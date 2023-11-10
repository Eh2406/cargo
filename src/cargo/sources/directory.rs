use crate::sources::source::Source;
use std::task::Poll;

pub struct DirectorySource<'cfg> {
    config: &'cfg (),
}

impl<'cfg> Source for DirectorySource<'cfg> {
    fn query(
        &mut self,
        dep: &Dependency,
    ) -> Poll<anyhow::Result<()>> {
        todo!()
    }
}
