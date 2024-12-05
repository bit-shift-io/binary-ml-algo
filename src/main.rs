use bitvec::prelude::*;

#[derive(Clone)]
pub struct Neuron {
    pub operator: u8, // todo: make an enum
    pub error_direction: u8
}

impl Neuron {
    fn new() -> Self {
        Self {
            operator: 0,
            error_direction: 0,
        }
    }
}


pub struct BinaryNN {
    pub connection_values: BitVec<u8, Msb0>,
    pub neurons: Vec<Neuron>,
}

impl BinaryNN {
    fn new() -> Self {
        let layer_1_len = 4;
        let layer_2_len = 2;
        let layer_3_len = 1;
        let total_len = layer_1_len + layer_2_len + layer_3_len;

        // https://docs.rs/bitvec/latest/bitvec/vec/struct.BitVec.html
        let mut connection_values = bitvec![u8, Msb0; 0; total_len];
        connection_values[0 .. total_len].store::<u8>(0x0);
        
        let neurons = vec![Neuron::new(); total_len];

        Self {
            connection_values,
            neurons
        }
    }

    fn set_input_values(&mut self, data: &[u8]) {
        assert_eq!(data.len(), 4); // todo: change this to layer_1_len

        for (i, v) in data.iter().enumerate() {
            let bool_val = (*v != 0);
            self.connection_values.set(i, bool_val);
        }
    }
}

fn main() {
    println!("Hello, world!");

    let mut bnn = BinaryNN::new();
    bnn.set_input_values(&[1, 0, 1, 0]);

    println!("done");
}
