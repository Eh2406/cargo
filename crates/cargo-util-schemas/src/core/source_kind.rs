/// The possible kinds of code source.
//
// Note that the ordering here is important for how it affects the `Ord`
// implementation, notably how this affects the ordering of serialized
// packages into lock files.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceKind {
    /// A local path.
    Path,
    /// A remote registry.
    Registry,
    /// A sparse registry.
    SparseRegistry,
    /// A local filesystem-based registry.
    LocalRegistry,
    /// A directory-based registry.
    Directory,
    /// A git repository.
    Git(GitReference),
}

impl SourceKind {
    pub fn protocol(&self) -> Option<&str> {
        match self {
            SourceKind::Path => Some("path"),
            SourceKind::Git(_) => Some("git"),
            SourceKind::Registry => Some("registry"),
            // Sparse registry URL already includes the `sparse+` prefix, see `SourceId::new`
            SourceKind::SparseRegistry => None,
            SourceKind::LocalRegistry => Some("local-registry"),
            SourceKind::Directory => Some("directory"),
        }
    }
}

/// Information to find a specific commit in a Git repository.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GitReference {
    /// From a tag.
    Tag(String),
    /// From a branch.
    Branch(String),
    /// From a specific revision. Can be a commit hash (either short or full),
    /// or a named reference like `refs/pull/493/head`.
    Rev(String),
    /// The default branch of the repository, the reference named `HEAD`.
    DefaultBranch,
}

impl GitReference {
    pub fn from_query(
        query_pairs: impl Iterator<Item = (impl AsRef<str>, impl AsRef<str>)>,
    ) -> Self {
        let mut reference = GitReference::DefaultBranch;
        for (k, v) in query_pairs {
            let v = v.as_ref();
            match k.as_ref() {
                // Map older 'ref' to branch.
                "branch" | "ref" => reference = GitReference::Branch(v.to_owned()),

                "rev" => reference = GitReference::Rev(v.to_owned()),
                "tag" => reference = GitReference::Tag(v.to_owned()),
                _ => {}
            }
        }
        reference
    }

    /// Returns a `Display`able view of this git reference, or None if using
    /// the head of the default branch
    pub fn pretty_ref(&self, url_encoded: bool) -> Option<PrettyRef<'_>> {
        match self {
            GitReference::DefaultBranch => None,
            _ => Some(PrettyRef {
                inner: self,
                url_encoded,
            }),
        }
    }
}

/// A git reference that can be `Display`ed
pub struct PrettyRef<'a> {
    inner: &'a GitReference,
    url_encoded: bool,
}

impl<'a> std::fmt::Display for PrettyRef<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value: &str;
        match self.inner {
            GitReference::Branch(s) => {
                write!(f, "branch=")?;
                value = s;
            }
            GitReference::Tag(s) => {
                write!(f, "tag=")?;
                value = s;
            }
            GitReference::Rev(s) => {
                write!(f, "rev=")?;
                value = s;
            }
            GitReference::DefaultBranch => unreachable!(),
        }
        if self.url_encoded {
            for value in url::form_urlencoded::byte_serialize(value.as_bytes()) {
                write!(f, "{value}")?;
            }
        } else {
            write!(f, "{value}")?;
        }
        Ok(())
    }
}
