use std::task::Poll;

pub trait Source {
    fn query(&mut self) -> Poll<anyhow::Result<impl Iterator<Item = ()>>>;
}

#[derive(Default)]
pub struct SourceMap<'src> {
    map: Box<dyn Source + 'src>,
}

impl<'src> SourceMap<'src> {
    pub fn get_mut(&mut self) -> Option<&mut (dyn Source + 'src)> {
        todo!();
    }
}