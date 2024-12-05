use bitvec::prelude::*;

pub struct BinaryNN {
    pub connection_values: BitVec<u8, Msb0>,
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
        
        Self {
            connection_values
        }
    }
}

fn main() {
    println!("Hello, world!");

    let bnn = BinaryNN::new();

    println!("done");
}
