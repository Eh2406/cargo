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

pub use super::encode::Metadata;
pub use super::encode::{EncodableDependency, EncodablePackageId, EncodableResolve};
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
pub type Activations =
    im_rc::HashMap<(InternedString, SourceId, SemverCompatibility), (Summary, ContextAge)>;

/// A type that represents when cargo treats two Versions as compatible.
/// Versions `a` and `b` are compatible if their left-most nonzero digit is the
/// same.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
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
    pub fn as_activations_key(self) -> (InternedString, SourceId, SemverCompatibility) {
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
    /// The `parent` passed in here is the parent summary/dependency edge which
    /// cased `summary` to get activated. This may not be present for the root
    /// crate, for example.
    ///
    /// Returns `true` if this summary with the given method is already activated.
    pub fn flag_activated(
        &mut self,
        summary: &Summary,
        method: &Method,
        parent: Option<(&Summary, &Dependency)>,
    ) -> CargoResult<bool> {
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

                // If we've got a parent dependency which activated us, *and*
                // the dependency has a different source id listed than the
                // `summary` itself, then things get interesting. This basically
                // means that a `[patch]` was used to augment `dep.source_id()`
                // with `summary`.
                //
                // In this scenario we want to consider the activation key, as
                // viewed from the perspective of `dep.source_id()`, as being
                // fulfilled. This means that we need to add a second entry in
                // the activations map for the source that was patched, in
                // addition to the source of the actual `summary` itself.
                //
                // Without this it would be possible to have both 1.0.0 and
                // 1.1.0 "from crates.io" in a dependency graph if one of those
                // versions came from a `[patch]` source.
                if let Some((_, dep)) = parent {
                    if dep.source_id() != id.source_id() {
                        let key = (id.name(), dep.source_id(), id.version().into());
                        let prev = self.activations.insert(key, (summary.clone(), age));
                        assert!(prev.is_none());
                    }
                }

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

    /// Checks whether all of `parent` and the keys of `conflicting activations`
    /// are still active.
    /// If so returns the `ContextAge` when the newest one was added.
    pub fn is_conflicting(
        &self,
        parent: Option<PackageId>,
        conflicting_activations: &ConflictMap,
    ) -> Option<usize> {
        let mut max = 0;
        for &id in conflicting_activations.keys().chain(parent.as_ref()) {
            if let Some(age) = self.is_active(id) {
                max = std::cmp::max(max, age);
            } else {
                return None;
            }
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

impl Graph<PackageId, Rc<Vec<(Dependency, ContextAge)>>> {
    pub fn parents_of(&self, p: PackageId) -> impl Iterator<Item = (PackageId, bool)> + '_ {
        self.edges(&p)
            .map(|(grand, d)| (*grand, d.iter().any(|(x, _)| x.is_public())))
    }
}

#[derive(Clone, Debug, Default)]
pub struct PublicDependency {
    /// For each active package the set of all the names it can see,
    /// for each name the exact package that name resolves to and whether it exports that visibility.
    inner: im_rc::HashMap<PackageId, im_rc::HashMap<InternedString, (PackageId, bool)>>,
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
            .filter(|x| x.1) // as publicly exported
            .map(|x| x.0)
            .chain(Some(candidate_pid)) // but even if not we know that everything exports itself
            .collect()
    }
    pub fn add_edge(
        &mut self,
        candidate_pid: PackageId,
        parent_pid: PackageId,
        is_public: bool,
        parents: &Graph<PackageId, Rc<Vec<(Dependency, ContextAge)>>>,
    ) {
        // one tricky part is that `candidate_pid` may already be active and
        // have public dependencies of its own. So we not only need to mark
        // `candidate_pid` as visible to its parents but also all of its existing
        // publicly exported dependencies.
        for c in self.publicly_exports(candidate_pid) {
            // for each (transitive) parent that can newly see `t`
            let mut stack = vec![(parent_pid, is_public)];
            while let Some((p, public)) = stack.pop() {
                match self.inner.entry(p).or_default().entry(c.name()) {
                    im_rc::hashmap::Entry::Occupied(mut o) => {
                        // the (transitive) parent can already see something by `c`s name, it had better be `c`.
                        assert_eq!(o.get().0, c);
                        if o.get().1 {
                            // The previous time the parent saw `c`, it was a public dependency.
                            // So all of its parents already know about `c`
                            // and we can save some time by stopping now.
                            continue;
                        }
                        if public {
                            // Mark that `c` has now bean seen publicly
                            o.insert((c, public));
                        }
                    }
                    im_rc::hashmap::Entry::Vacant(v) => {
                        // The (transitive) parent does not have anything by `c`s name,
                        // so we add `c`.
                        v.insert((c, public));
                    }
                }
                // if `candidate_pid` was a private dependency of `p` then `p` parents can't see `c` thru `p`
                if public {
                    // if it was public, then we add all of `p`s parents to be checked
                    stack.extend(parents.parents_of(p));
                }
            }
        }
    }
    pub fn can_add_edge(
        &self,
        b_id: PackageId,
        parent: PackageId,
        is_public: bool,
        parents: &Graph<PackageId, Rc<Vec<(Dependency, ContextAge)>>>,
    ) -> Result<(), (PackageId, ConflictReason)> {
        // one tricky part is that `candidate_pid` may already be active and
        // have public dependencies of its own. So we not only need to check
        // `b_id` as visible to its parents but also all of its existing
        // publicly exported dependencies.
        for t in self.publicly_exports(b_id) {
            // for each (transitive) parent that can newly see `t`
            let mut stack = vec![(parent, is_public)];
            while let Some((p, public)) = stack.pop() {
                // TODO: dont look at the same thing more then once
                if let Some(o) = self.inner.get(&p).and_then(|x| x.get(&t.name())) {
                    if o.0 != t {
                        // the (transitive) parent can already see a different version by `t`s name.
                        // So, adding `b` will cause `p` to have a public dependency conflict on `t`.
                        return Err((p, ConflictReason::PublicDependency));
                    }
                    if o.1 {
                        // The previous time the parent saw `t`, it was a public dependency.
                        // So all of its parents already know about `t`
                        // and we can save some time by stopping now.
                        continue;
                    }
                }
                // if `b` was a private dependency of `p` then `p` parents can't see `t` thru `p`
                if public {
                    // if it was public, then we add all of `p`s parents to be checked
                    stack.extend(parents.parents_of(p));
                }
            }
        }
        Ok(())
    }
}
