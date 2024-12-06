use std::{collections::VecDeque, fmt};

use bitvec::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrainResult {
    Success,
    Failure,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operator {
    And,
    Or,
    Not,
    Xor,
    Zero,
    One,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
        // or, alternatively:
        // fmt::Debug::fmt(self, f)
    }
}

/* 
#[derive(Clone)]
pub struct Operator {
    pub eoperator: EOperator,
}*/

impl Operator {
    fn new() -> Self {
        Operator::And
    }

    fn apply(&self, a: u8, b: u8) -> u8 {
        use Operator::*;
        match *self {
            And => a & b,
            Or => a | b,
            Not =>!a,
            Xor => a ^ b,
            Zero => 0,
            One => 0xFF,
        }
    }

    fn get_next_operator(&self) -> Self {
        use Operator::*;
        match *self {
            And => Or,
            Or => Not,
            Not => Xor,
            Xor => Zero,
            Zero => One,
            One => And,
        }
    }

    fn is_last(&self) -> bool {
        use Operator::*;
        match *self {
            One => true,
            _ => false,
        }
    }
}

#[derive(Clone)]
pub struct Neuron {
    pub operator: Operator,
    pub error_direction: u8
}

impl Neuron {
    fn new() -> Self {
        Self {
            operator: Operator::new(),
            error_direction: 0,
        }
    }
}


pub struct BinaryNN {
    pub connection_values: BitVec<u8, Msb0>,
    pub neurons: Vec<Neuron>,

    // This contains indicies of the last changed neuron indexes
    // to help us determine which neuron to change next
    pub neuron_error_indexes: VecDeque<usize>
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
        
        let layer_1_neurons_len = 2;
        let layer_2_neuron_len = 1;
        let total_neuron_len = layer_1_neurons_len + layer_2_neuron_len;
        let neurons = vec![Neuron::new(); total_neuron_len];

        Self {
            connection_values,
            neurons,
            neuron_error_indexes: VecDeque::new(),
        }
    }

    fn set_input_values(&mut self, data: &[u8]) {
        assert_eq!(data.len(), 4); // todo: change this to layer_1_len

        for (i, v) in data.iter().enumerate() {
            let bool_val = *v != 0;
            self.connection_values.set(i, bool_val);
        }
    }

    fn train(&mut self, inputs: &[u8], outputs: &[u8], max_steps: usize) -> TrainResult {
        // loop until train_step returns false or max_steps is reached
        for _ in 0..max_steps {
            let train_result = self.train_step(inputs, outputs);
            if train_result == TrainResult::Success {
                return TrainResult::Success;
            }
        }

        return TrainResult::Failure;
    }

    fn train_step(&mut self, inputs: &[u8], outputs: &[u8]) -> TrainResult {
        self.forward_propagate(inputs);

        let error = self.check_for_error(outputs);

        // back propagate on error, starting at the last neuron
        if error == TrainResult::Failure {
            self.back_propagate(self.neurons.len() - 1);
        }

        return error;
    }

    fn forward_propagate(&mut self, inputs: &[u8]) {
        self.set_input_values(inputs);

        let layer_lens = [4, 2]; //, 1]; - we only need to iterate over n-1 layers, as n-1 computes the results for layer n
        

        // feed forward
        // 1. iterate over each layer
        // 2. iterate over each neuron in the layer, compute the result and push to the next layers input
        let mut i = 0;
        let mut start_of_layer_index = 0;
        let mut start_of_next_layer_index = 0;
        for (_, layer_len) in layer_lens.iter().enumerate() {
            start_of_next_layer_index += layer_len;

            for j in (0..*layer_len).step_by(2) {
                let a = if self.connection_values[i] { 0x1 } else { 0x0 };
                let b = if self.connection_values[i + 1] { 0x1 } else { 0x0 };

                let ni = i / 2;
                let output = self.neurons[ni].operator.apply(a, b);
                println!("n{}:  a:{} b:{} -> op:{} -> o:{}", ni, a, b, self.neurons[ni].operator, output);

                // push the results to the inputs for the next layer
                let neuron_index_on_current_layer = j / 2;
                let out_i = start_of_next_layer_index + neuron_index_on_current_layer;
                self.connection_values.set(out_i, output != 0);

                i += 2;
            }

            start_of_layer_index += layer_len;
        }
    }

    fn check_for_error(&self, outputs: &[u8]) -> TrainResult {
        let layer_lens = [4, 2]; //, 1]; - we only need to iterate over n-1 layers, as n-1 computes the results for layer n
        let last_layer_len = 1;

        let mut i = 0;
        for (_, layer_len) in layer_lens.iter().enumerate() {
            i += layer_len;
        }

        // compare results with outputs to find an error value
        // 1. iterate over the last layer
        let mut error = TrainResult::Success;
        for _ in 0..last_layer_len {
            let o = if self.connection_values[i] { 0x1 } else { 0x0 };

            if outputs[0] != o {
                error = TrainResult::Failure;
                println!("Error. Expected {} but got {}", outputs[0], o);
                break;
            }

            i += 1;
        }

        if error == TrainResult::Success {
            println!("Success");
        }

        return error;
    }

    fn back_propagate(&mut self, ni: usize) {
        
        // change the top level neuron first
        // then on consecutive errors, feed them to our children
        //

        // find the neuron that needs to change, this could just be an index
        // pointing to the last neuron idx that changed and decrement it?
        let mut ni = *self.neuron_error_indexes.iter().min().or(Some(&self.neurons.len())).unwrap();
        if ni == self.neurons.len() {
            self.neuron_error_indexes.clear();
            ni = self.neurons.len() - 1;
        }

        self.neurons[ni].operator = self.neurons[ni].operator.get_next_operator();

        self.neuron_error_indexes.push_back(ni);

        // todo: I don't think this will work to simply try all changes and use the first that will work for this neuron
        // instead we should change this neuron till we read is_last, then pass to the children till they all reach is_last
        // then again switch to changing this neuron...


        /*
        // change the neurons operator, seeing if any of them change the output to the correct results
        let start_operator = self.neurons[ni].operator.clone();
        self.neurons[ni].operator = self.neurons[ni].operator.get_next_operator();

        while self.neurons[ni].operator != start_operator {
           
            // check if the new operator resolves the error
            let is_error_resolved = false;
            if is_error_resolved {
                return;
            }

            // try the next operator
            self.neurons[ni].operator = self.neurons[ni].operator.get_next_operator();
        }

        // changing the operator did not resolve the problem,
        // so we need to try forwarding the error to each of our children until something they do results in the correct result
        self.neurons[ni].error_direction += 1;

        if ni == 0 {
            return; // we've reached the end, no more neurons to back propagate to
        }

        return self.back_propagate(ni - 1);
        */
    }
}


fn example_function_to_learn(data: &[u8]) -> u8 {
    let n1 = data[0] | data[1];
    let n2 = data[2] & data[3];
    let n3 = n1 | n2;
    return n3;
}

fn main() {
    println!("Binary Machine Learning Algorithm");

    let max_steps = 100;
    let mut bnn = BinaryNN::new();

    // all possible permutations for the function inputs
    let function_inputs = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ];

    // run the function over the inputs to generate a tuple of (input, output) pairs
    let training_data = function_inputs.map(|inputs| {
        return (inputs, [example_function_to_learn(&inputs)]);
    });

    // learn on training data
    for (inputs, outputs) in training_data {
        let train_result = bnn.train(&inputs, &outputs, max_steps);
        match train_result {
            TrainResult::Failure => println!("Failed to learn!"),
            TrainResult::Success => println!("Succesfully Learned!"),
        }
    }
    /*
    let in_1 = [1, 0, 0, 0];
    let out_1 = [example_function_to_learn(&in_1)];
    let train_result = bnn.train(&in_1, &out_1, max_steps);

    match train_result {
        TrainResult::Failure => println!("Failed to learn!"),
        TrainResult::Success => println!("Succesfully Learned!"),
    }
*/
    println!("done");
}
