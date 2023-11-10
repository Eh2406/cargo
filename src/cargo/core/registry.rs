use std::task::Poll;

use crate::sources::source::{Source, SourceMap};

pub trait Registry {
    fn query(&mut self) -> Poll<anyhow::Result<impl Iterator<Item = ()>>>;
}
pub struct PackageRegistry<'cfg> {
    sources: SourceMap<'cfg>,
}

impl<'cfg> Registry for PackageRegistry<'cfg> {
    fn query(&mut self) -> Poll<anyhow::Result<impl Iterator<Item = ()>>> {
        self.sources.get_mut();
        todo!();
        Poll::Ready(Ok([].into_iter()))
    }
}

fn summary_for_patch(source: &mut dyn Source) -> Poll<anyhow::Result<()>> {
    source.query();
    todo!();
}
