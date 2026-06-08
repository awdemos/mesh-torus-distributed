#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TorusCoordinates {
    pub x: usize,
    pub y: usize,
    pub z: Option<usize>,
}

impl TorusCoordinates {
    pub fn new_2d(x: usize, y: usize) -> Self {
        Self {
            x,
            y,
            z: None,
        }
    }

    pub fn new_3d(x: usize, y: usize, z: usize) -> Self {
        Self {
            x,
            y,
            z: Some(z),
        }
    }

    pub fn is_3d(&self) -> bool {
        self.z.is_some()
    }

    pub fn linear_index(&self, dims: &TorusDimensions) -> usize {
        match self.z {
            Some(z) => z * dims.width * dims.height + self.y * dims.width + self.x,
            None => self.y * dims.width + self.x,
        }
    }

    pub fn from_linear(index: usize, dims: &TorusDimensions) -> Self {
        match dims.depth {
            Some(depth) => {
                let z = index / (dims.width * dims.height);
                let rem = index % (dims.width * dims.height);
                let y = rem / dims.width;
                let x = rem % dims.width;
                Self::new_3d(x, y, z % depth)
            }
            None => {
                let y = index / dims.width;
                let x = index % dims.width;
                Self::new_2d(x, y)
            }
        }
    }

    pub fn neighbors(&self, dims: &TorusDimensions) -> Vec<(TorusCoordinates, usize)> {
        let mut result = Vec::new();
        let axes = if self.z.is_some() { 3 } else { 2 };
        for axis in 0..axes {
            result.push((self.neighbor(dims, axis, true), axis));
            result.push((self.neighbor(dims, axis, false), axis));
        }
        result
    }

    pub fn neighbor(&self, dims: &TorusDimensions, axis: usize, positive: bool) -> TorusCoordinates {
        match axis {
            0 => {
                let x = wrap_neighbor(self.x, dims.width, positive);
                TorusCoordinates { x, y: self.y, z: self.z }
            }
            1 => {
                let y = wrap_neighbor(self.y, dims.height, positive);
                TorusCoordinates { x: self.x, y, z: self.z }
            }
            2 => {
                let depth = dims.depth.unwrap_or(1);
                let z_val = self.z.unwrap_or(0);
                let z = wrap_neighbor(z_val, depth, positive);
                TorusCoordinates { x: self.x, y: self.y, z: Some(z) }
            }
            _ => panic!("axis out of range: {}", axis),
        }
    }
}

fn wrap_neighbor(current: usize, size: usize, positive: bool) -> usize {
    if positive {
        (current + 1) % size
    } else {
        (current + size - 1) % size
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TorusDimensions {
    pub width: usize,
    pub height: usize,
    pub depth: Option<usize>,
}

impl TorusDimensions {
    pub fn new_2d(width: usize, height: usize) -> Self {
        Self { width, height, depth: None }
    }

    pub fn new_3d(width: usize, height: usize, depth: usize) -> Self {
        Self { width, height, depth: Some(depth) }
    }

    pub fn total_nodes(&self) -> usize {
        match self.depth {
            Some(d) => self.width * self.height * d,
            None => self.width * self.height,
        }
    }

    pub fn axis_count(&self) -> usize {
        if self.depth.is_some() { 3 } else { 2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_2d_coordinates() {
        let c = TorusCoordinates::new_2d(1, 2);
        assert_eq!(c.x, 1);
        assert_eq!(c.y, 2);
        assert!(c.z.is_none());
        assert!(!c.is_3d());
    }

    #[rstest]
    fn test_3d_coordinates() {
        let c = TorusCoordinates::new_3d(1, 2, 3);
        assert_eq!(c.x, 1);
        assert_eq!(c.y, 2);
        assert_eq!(c.z, Some(3));
        assert!(c.is_3d());
    }

    #[rstest]
    fn test_2d_linear_index() {
        let dims = TorusDimensions::new_2d(4, 3);
        let c = TorusCoordinates::new_2d(2, 1);
        assert_eq!(c.linear_index(&dims), 6);
    }

    #[rstest]
    fn test_3d_linear_index() {
        let dims = TorusDimensions::new_3d(4, 3, 2);
        let c = TorusCoordinates::new_3d(2, 1, 1);
        assert_eq!(c.linear_index(&dims), 18);
    }

    #[rstest]
    fn test_from_linear_2d() {
        let dims = TorusDimensions::new_2d(4, 3);
        let c = TorusCoordinates::from_linear(6, &dims);
        assert_eq!(c, TorusCoordinates::new_2d(2, 1));
    }

    #[rstest]
    fn test_from_linear_3d() {
        let dims = TorusDimensions::new_3d(4, 3, 2);
        let c = TorusCoordinates::from_linear(18, &dims);
        assert_eq!(c, TorusCoordinates::new_3d(2, 1, 1));
    }

    #[rstest]
    fn test_2d_neighbors_interior() {
        let dims = TorusDimensions::new_2d(4, 4);
        let c = TorusCoordinates::new_2d(1, 1);
        let neighbors = c.neighbors(&dims);
        assert_eq!(neighbors.len(), 4);
        let coords: Vec<_> = neighbors.into_iter().map(|(coord, _)| coord).collect();
        assert!(coords.contains(&TorusCoordinates::new_2d(2, 1)));
        assert!(coords.contains(&TorusCoordinates::new_2d(0, 1)));
        assert!(coords.contains(&TorusCoordinates::new_2d(1, 2)));
        assert!(coords.contains(&TorusCoordinates::new_2d(1, 0)));
    }

    #[rstest]
    fn test_2d_neighbors_wrap() {
        let dims = TorusDimensions::new_2d(3, 3);
        let c = TorusCoordinates::new_2d(0, 0);
        let right = c.neighbor(&dims, 0, true);
        let left = c.neighbor(&dims, 0, false);
        let down = c.neighbor(&dims, 1, true);
        let up = c.neighbor(&dims, 1, false);
        assert_eq!(right, TorusCoordinates::new_2d(1, 0));
        assert_eq!(left, TorusCoordinates::new_2d(2, 0));
        assert_eq!(down, TorusCoordinates::new_2d(0, 1));
        assert_eq!(up, TorusCoordinates::new_2d(0, 2));
    }

    #[rstest]
    fn test_3d_neighbors() {
        let dims = TorusDimensions::new_3d(3, 3, 3);
        let c = TorusCoordinates::new_3d(1, 1, 1);
        let neighbors = c.neighbors(&dims);
        assert_eq!(neighbors.len(), 6);
        let coords: Vec<_> = neighbors.into_iter().map(|(coord, _)| coord).collect();
        assert!(coords.contains(&TorusCoordinates::new_3d(2, 1, 1)));
        assert!(coords.contains(&TorusCoordinates::new_3d(0, 1, 1)));
        assert!(coords.contains(&TorusCoordinates::new_3d(1, 2, 1)));
        assert!(coords.contains(&TorusCoordinates::new_3d(1, 0, 1)));
        assert!(coords.contains(&TorusCoordinates::new_3d(1, 1, 2)));
        assert!(coords.contains(&TorusCoordinates::new_3d(1, 1, 0)));
    }

    #[rstest]
    fn test_3d_neighbors_wrap() {
        let dims = TorusDimensions::new_3d(3, 3, 3);
        let c = TorusCoordinates::new_3d(0, 0, 0);
        assert_eq!(
            c.neighbor(&dims, 0, false),
            TorusCoordinates::new_3d(2, 0, 0)
        );
        assert_eq!(
            c.neighbor(&dims, 1, false),
            TorusCoordinates::new_3d(0, 2, 0)
        );
        assert_eq!(
            c.neighbor(&dims, 2, false),
            TorusCoordinates::new_3d(0, 0, 2)
        );
        assert_eq!(
            c.neighbor(&dims, 2, true),
            TorusCoordinates::new_3d(0, 0, 1)
        );
    }

    #[rstest]
    fn test_dimensions_total_nodes_2d() {
        let dims = TorusDimensions::new_2d(4, 3);
        assert_eq!(dims.total_nodes(), 12);
    }

    #[rstest]
    fn test_dimensions_total_nodes_3d() {
        let dims = TorusDimensions::new_3d(4, 3, 2);
        assert_eq!(dims.total_nodes(), 24);
    }

    #[rstest]
    fn test_dimensions_axis_count() {
        assert_eq!(TorusDimensions::new_2d(4, 3).axis_count(), 2);
        assert_eq!(TorusDimensions::new_3d(4, 3, 2).axis_count(), 3);
    }
}
