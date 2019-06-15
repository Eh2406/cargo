use std::collections::HashMap;
use std::num::NonZeroU64;
use std::rc::Rc;

// "ensure" seems to require "bail" be in scope (macro hygiene issue?).
#[allow(unused_imports)]
use failure::{bail, ensure};
use log::debug;

use crate::core::interning::InternedString;
use crate::core::{Dependency, PackageId, SourceId, Summary};
use crate::util::CargoResult;
use crate::util::Graph;

use super::dep_cache::RegistryQueryer;
use super::types::{ConflictMap, ConflictReason, FeaturesSet, Method};

pub use super::encode::{EncodableDependency, EncodablePackageId, EncodableResolve};
pub use super::encode::{Metadata, WorkspaceResolve};
pub use super::resolve::Resolve;

// A `Context` is basically a bunch of local resolution information which is
// kept around for all `BacktrackFrame` instances. As a result, this runs the
// risk of being cloned *a lot* so we want to make this as cheap to clone as
// possible.
#[derive(Clone)]
pub struct Context {
    pub age: ContextAge,
    pub activations: Activations,
    /// list the features that are activated for each package
    pub resolve_features: im_rc::HashMap<PackageId, FeaturesSet>,
    /// get the package that will be linking to a native library by its links attribute
    pub links: im_rc::HashMap<InternedString, PackageId>,
    /// for each package the list of names it can see,
    /// then for each name the exact version that name represents and weather the name is public.
    pub public_dependency: Option<PublicDependency>,

    /// a way to look up for a package in activations what packages required it
    /// and all of the exact deps that it fulfilled.
    pub parents: Graph<PackageId, Rc<Vec<(Dependency, ContextAge)>>>,
}

/// When backtracking it can be useful to know how far back to go.
/// The `ContextAge` of a `Context` is a monotonically increasing counter of the number
/// of decisions made to get to this state.
/// Several structures store the `ContextAge` when it was added,
/// to be used in `find_candidate` for backtracking.
pub type ContextAge = usize;

/// Find the activated version of a crate based on the name, source, and semver compatibility.
/// By storing this in a hash map we ensure that there is only one
/// semver compatible version of each crate.
/// This all so stores the `ContextAge`.
pub type ActivationsKey = (InternedString, SourceId, SemverCompatibility);
pub type Activations = im_rc::HashMap<ActivationsKey, (Summary, ContextAge)>;

/// A type that represents when cargo treats two Versions as compatible.
/// Versions `a` and `b` are compatible if their left-most nonzero digit is the
/// same.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
pub enum SemverCompatibility {
    Major(NonZeroU64),
    Minor(NonZeroU64),
    Patch(u64),
}

impl From<&semver::Version> for SemverCompatibility {
    fn from(ver: &semver::Version) -> Self {
        if let Some(m) = NonZeroU64::new(ver.major) {
            return SemverCompatibility::Major(m);
        }
        if let Some(m) = NonZeroU64::new(ver.minor) {
            return SemverCompatibility::Minor(m);
        }
        SemverCompatibility::Patch(ver.patch)
    }
}

impl PackageId {
    pub fn as_activations_key(self) -> ActivationsKey {
        (self.name(), self.source_id(), self.version().into())
    }
}

impl Context {
    pub fn new(check_public_visible_dependencies: bool) -> Context {
        Context {
            age: 0,
            resolve_features: im_rc::HashMap::new(),
            links: im_rc::HashMap::new(),
            public_dependency: if check_public_visible_dependencies {
                Some(PublicDependency::new())
            } else {
                None
            },
            parents: Graph::new(),
            activations: im_rc::HashMap::new(),
        }
    }

    /// Activate this summary by inserting it into our list of known activations.
    ///
    /// Returns `true` if this summary with the given method is already activated.
    pub fn flag_activated(&mut self, summary: &Summary, method: &Method) -> CargoResult<bool> {
        let id = summary.package_id();
        let age: ContextAge = self.age;
        match self.activations.entry(id.as_activations_key()) {
            im_rc::hashmap::Entry::Occupied(o) => {
                debug_assert_eq!(
                    &o.get().0,
                    summary,
                    "cargo does not allow two semver compatible versions"
                );
            }
            im_rc::hashmap::Entry::Vacant(v) => {
                if let Some(link) = summary.links() {
                    ensure!(
                        self.links.insert(link, id).is_none(),
                        "Attempting to resolve a dependency with more then one crate with the \
                         links={}.\nThis will not build as is. Consider rebuilding the .lock file.",
                        &*link
                    );
                }
                v.insert((summary.clone(), age));
                return Ok(false);
            }
        }
        debug!("checking if {} is already activated", summary.package_id());
        let (features, use_default) = match method {
            Method::Everything
            | Method::Required {
                all_features: true, ..
            } => return Ok(false),
            Method::Required {
                features,
                uses_default_features,
                ..
            } => (features, uses_default_features),
        };

        let has_default_feature = summary.features().contains_key("default");
        Ok(match self.resolve_features.get(&id) {
            Some(prev) => {
                features.is_subset(prev)
                    && (!use_default || prev.contains("default") || !has_default_feature)
            }
            None => features.is_empty() && (!use_default || !has_default_feature),
        })
    }

    /// If the package is active returns the `ContextAge` when it was added
    pub fn is_active(&self, id: PackageId) -> Option<ContextAge> {
        self.activations
            .get(&id.as_activations_key())
            .and_then(|(s, l)| if s.package_id() == id { Some(*l) } else { None })
    }

    /// If the conflict reason on the package still applies returns the `ContextAge` when it was added
    pub fn still_applies(&self, id: PackageId, reason: &ConflictReason) -> Option<ContextAge> {
        self.is_active(id).and_then(|mut max| {
            match reason {
                ConflictReason::PublicDependency(name, is_public) => {
                    max = std::cmp::max(max, self.is_active(*name)?);
                    max = std::cmp::max(
                        max,
                        self.public_dependency.as_ref().unwrap().can_add_item_b(
                            id,
                            *name,
                            *is_public,
                            &self.parents,
                        )?,
                    );
                }
                ConflictReason::PubliclyExports(name) => {
                    if &id == name {
                        return Some(max);
                    }
                    max = std::cmp::max(max, self.is_active(*name)?);
                    max = std::cmp::max(
                        max,
                        self.public_dependency
                            .as_ref()
                            .unwrap()
                            .publicly_exports_item(*name, id)?,
                    );
                }
                _ => {}
            }
            Some(max)
        })
    }

    /// Checks whether all of `parent` and the keys of `conflicting activations`
    /// are still active.
    /// If so returns the `ContextAge` when the newest one was added.
    pub fn is_conflicting(
        &self,
        parent: Option<PackageId>,
        conflicting_activations: &ConflictMap,
    ) -> Option<usize> {
        let mut max = 0;
        if let Some(parent) = parent {
            max = std::cmp::max(max, self.is_active(parent)?);
        }

        for (id, reason) in conflicting_activations.iter() {
            max = std::cmp::max(max, self.still_applies(*id, reason)?);
        }
        Some(max)
    }

    pub fn resolve_replacements(
        &self,
        registry: &RegistryQueryer<'_>,
    ) -> HashMap<PackageId, PackageId> {
        self.activations
            .values()
            .filter_map(|(s, _)| registry.used_replacement_for(s.package_id()))
            .collect()
    }

    pub fn graph(&self) -> Graph<PackageId, Vec<Dependency>> {
        let mut graph: Graph<PackageId, Vec<Dependency>> = Graph::new();
        self.activations
            .values()
            .for_each(|(r, _)| graph.add(r.package_id()));
        for i in self.parents.iter() {
            graph.add(*i);
            for (o, e) in self.parents.edges(i) {
                let old_link = graph.link(*o, *i);
                assert!(old_link.is_empty());
                *old_link = e.iter().map(|(d, _)| d.clone()).collect();
            }
        }
        graph
    }
}

#[derive(Copy, Clone, Debug)]
enum PublicContextAge {
    /// This is only exported publicly, or this was exported publicly originally
    JustPublic(ContextAge),
    /// This was exported privately, and then later upgraded to exported publicly
    Both(ContextAge, ContextAge),
    /// This is only exported privately
    JustPrivate(ContextAge),
}

impl PublicContextAge {
    fn is_public(&self) -> bool {
        if let PublicContextAge::JustPrivate(_) = self {
            return false;
        }
        true
    }
    fn public_age(&self) -> Option<ContextAge> {
        match self {
            PublicContextAge::JustPublic(a) => Some(*a),
            PublicContextAge::Both(_, a) => Some(*a),
            PublicContextAge::JustPrivate(_) => None,
        }
    }
    fn private_age(&self) -> ContextAge {
        match self {
            PublicContextAge::JustPublic(a) => *a,
            PublicContextAge::Both(a, _) => *a,
            PublicContextAge::JustPrivate(a) => *a,
        }
    }
}

fn as_public_context_age(deps: &[(Dependency, ContextAge)]) -> PublicContextAge {
    if cfg!(debug_assertions) {
        // deps are sorted by age
        let mut max = 0;
        for (_, age) in deps.iter() {
            assert!(max <= *age);
            max = *age;
        }
    }
    let mut priv_age = None;
    for (dep, age) in deps.iter() {
        if dep.is_public() {
            if let Some(priv_age) = priv_age {
                return PublicContextAge::Both(priv_age, *age);
            }
            return PublicContextAge::JustPublic(*age);
        }
        priv_age.get_or_insert(*age);
    }
    PublicContextAge::JustPrivate(priv_age.expect("dep list must not be empty"))
}

impl Graph<PackageId, Rc<Vec<(Dependency, ContextAge)>>> {
    fn parents_of(&self, p: PackageId) -> impl Iterator<Item = (PackageId, PublicContextAge)> + '_ {
        self.edges(&p)
            .map(|(grand, d)| (*grand, as_public_context_age(d)))
    }
}

#[derive(Clone, Debug, Default)]
pub struct PublicDependency {
    /// For each active package the set of all the names it can see,
    /// for each name the exact package that name resolves to,
    ///     the `ContextAge` when it was first visible,
    ///     and whether it exports that visibility.
    inner: im_rc::HashMap<PackageId, im_rc::HashMap<InternedString, (PackageId, PublicContextAge)>>,
}

impl PublicDependency {
    fn new() -> Self {
        PublicDependency {
            inner: im_rc::HashMap::new(),
        }
    }
    fn publicly_exports(&self, candidate_pid: PackageId) -> Vec<PackageId> {
        self.inner
            .get(&candidate_pid) // if we have seen it before
            .iter()
            .flat_map(|x| x.values()) // all the things we have stored
            .filter(|x| x.1.is_public()) // as publicly exported
            .map(|x| x.0)
            .chain(Some(candidate_pid)) // but even if not we know that everything exports itself
            .collect()
    }
    fn publicly_exports_item(
        &self,
        candidate_pid: PackageId,
        target: PackageId,
    ) -> Option<ContextAge> {
        debug_assert_ne!(candidate_pid, target);
        let out = self
            .inner
            .get(&candidate_pid)
            .and_then(|names| names.get(&target.name()))
            .filter(|(p, _)| *p == target)
            .and_then(|(_, age)| age.public_age());
        debug_assert_eq!(
            out.is_some(),
            self.publicly_exports(candidate_pid).contains(&target)
        );
        out
    }
    pub fn add_edge(
        &mut self,
        candidate_pid: PackageId,
        parent_pid: PackageId,
        is_public: bool,
        age: ContextAge,
        parents: &Graph<PackageId, Rc<Vec<(Dependency, ContextAge)>>>,
    ) {
        // one tricky part is that `candidate_pid` may already be active and
        // have public dependencies of its own. So we not only need to mark
        // `candidate_pid` as visible to its parents but also all of its existing
        // publicly exported dependencies.
        let age = if is_public {
            PublicContextAge::JustPublic(age)
        } else {
            PublicContextAge::JustPrivate(age)
        };
        for c in self.publicly_exports(candidate_pid) {
            // for each (transitive) parent that can newly see `t`
            let mut stack = vec![(parent_pid, age)];
            while let Some((p, public)) = stack.pop() {
                match self.inner.entry(p).or_default().entry(c.name()) {
                    im_rc::hashmap::Entry::Occupied(mut o) => {
                        // the (transitive) parent can already see something by `c`s name, it had better be `c`.
                        assert_eq!(o.get().0, c);
                        if o.get().1.is_public() {
                            // The previous time the parent saw `c`, it was a public dependency.
                            // So all of its parents already know about `c`
                            // and we can save some time by stopping now.
                            continue;
                        }
                        if let Some(public_age) = age.public_age() {
                            // Mark that `c` has now bean seen publicly
                            let old_age = o.get().1.private_age();
                            o.insert((c, PublicContextAge::Both(old_age, public_age)));
                        }
                    }
                    im_rc::hashmap::Entry::Vacant(v) => {
                        // The (transitive) parent does not have anything by `c`s name,
                        // so we add `c`.
                        if public.is_public() {
                            v.insert((c, age));
                        } else {
                            v.insert((c, PublicContextAge::JustPrivate(age.private_age())));
                        }
                    }
                }
                // if `candidate_pid` was a private dependency of `p` then `p` parents can't see `c` thru `p`
                if public.is_public() {
                    // if it was public, then we add all of `p`s parents to be checked
                    stack.extend(parents.parents_of(p));
                }
            }
        }
        self.inner
            .entry(candidate_pid)
            .or_default()
            .entry(candidate_pid.name())
            .or_insert((
                candidate_pid,
                PublicContextAge::JustPrivate(age.private_age()),
            ));
    }
    pub fn can_add_edge(
        &self,
        b_id: PackageId,
        parent: PackageId,
        is_public: bool,
        parents: &Graph<PackageId, Rc<Vec<(Dependency, ContextAge)>>>,
    ) -> Result<
        (),
        (
            (PackageId, ConflictReason),
            Option<(PackageId, ConflictReason)>,
        ),
    > {
        // one tricky part is that `candidate_pid` may already be active and
        // have public dependencies of its own. So we not only need to check
        // `b_id` as visible to its parents but also all of its existing
        // publicly exported dependencies.
        for t in self.publicly_exports(b_id) {
            self.can_add_item_a(t, parent, is_public, parents)
                .map_err(|e| {
                    if t == b_id {
                        (e, None)
                    } else {
                        (e, Some((t, ConflictReason::PubliclyExports(b_id))))
                    }
                })?;
        }
        Ok(())
    }
    pub fn can_add_item_a(
        &self,
        t: PackageId,
        parent: PackageId,
        is_public: bool,
        parents: &Graph<PackageId, Rc<Vec<(Dependency, ContextAge)>>>,
    ) -> Result<(), (PackageId, ConflictReason)> {
        let mut stack = vec![(
            parent,
            if is_public {
                PublicContextAge::JustPublic(0)
            } else {
                PublicContextAge::JustPrivate(0)
            },
        )];
        // for each (transitive) parent that can newly see `t`
        while let Some((p, public)) = stack.pop() {
            // TODO: dont look at the same thing more then once
            if let Some(o) = self.inner.get(&p).and_then(|x| x.get(&t.name())) {
                if o.0 != t {
                    // the (transitive) parent can already see a different version by `t`s name.
                    // So, adding `b` will cause `p` to have a public dependency conflict on `t`.
                    return Err((o.0, ConflictReason::PublicDependency(parent, is_public)));
                }
            }
            // if `b` was a private dependency of `p` then `p` parents can't see `t` thru `p`
            if public.is_public() {
                // if it was public, then we add all of `p`s parents to be checked
                stack.extend(parents.parents_of(p));
            }
        }
        Ok(())
    }
    pub fn can_add_item_b(
        &self,
        t: PackageId,
        parent: PackageId,
        is_public: bool,
        parents: &Graph<PackageId, Rc<Vec<(Dependency, ContextAge)>>>,
    ) -> Option<ContextAge> {
        let mut is_constrained = None;
        let mut stack = vec![(
            0,
            (
                parent,
                if is_public {
                    PublicContextAge::JustPublic(0)
                } else {
                    PublicContextAge::JustPrivate(0)
                },
            ),
        )];
        // for each (transitive) parent that can newly see `t`
        while let Some((path_age, (p, public))) = stack.pop() {
            // TODO: dont look at the same thing more then once
            if let Some(o) = self.inner.get(&p).and_then(|x| x.get(&t.name())) {
                if o.0 == t {
                    let path_age = std::cmp::max(path_age, public.private_age());
                    let total_age = std::cmp::max(path_age, o.1.private_age());
                    if *is_constrained.get_or_insert(total_age) > total_age {
                        // we found one that can jump-back further so replace the out.
                        is_constrained = Some(total_age);
                    }
                }
            }
            // if `b` was a private dependency of `p` then `p` parents can't see `t` thru `p`
            if let Some(public_age) = public.public_age() {
                let path_age = std::cmp::max(path_age, public_age);
                // if it was public, then we add all of `p`s parents to be checked
                stack.extend(parents.parents_of(p).map(|g| (path_age, g)));
            }
        }
        is_constrained
    }
}
