use std::{collections::VecDeque, fmt};
use rand::seq::SliceRandom; // Import for shuffle
use rand::thread_rng;
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

impl Operator {
    fn new() -> Self {
        Operator::And
    }

    fn apply(&self, a: u8, b: u8) -> u8 {
        use Operator::*;
        let r = match *self {
            And => a & b,
            Or => a | b,
            Not =>!a,
            Xor => a ^ b,
            Zero => 0x0,
            One => 0x1,
        };
        return r & 0x1; // we only care about the first bit
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

    // Returns a pair containing (final result, how many steps were taken)
    fn train(&mut self, inputs: &[u8], outputs: &[u8], max_steps: usize) -> (TrainResult, usize) {
        // loop until train_step returns false or max_steps is reached
        let mut error_count = 0;
        for _ in 0..max_steps {
            let train_result = self.train_step(inputs, outputs);
            if train_result == TrainResult::Success {
                return (TrainResult::Success, error_count);
            } else {
                error_count += 1;
            }
        }

        return (TrainResult::Failure, error_count);
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
        if ni == 0 {
            self.neuron_error_indexes.clear();
            ni = self.neurons.len() - 1;
        } else {
            ni = ni - 1;
        }

        let previous_neuron_operator = self.neurons[ni].operator.clone();
        self.neurons[ni].operator = self.neurons[ni].operator.get_next_operator();
        println!("Changing n{} from {} -> {}", ni, previous_neuron_operator, self.neurons[ni].operator);
        println!("");

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

fn example_function_to_learn_2(data: &[u8]) -> u8 {
    let n1 = data[0] | data[1];
    let n2 = data[2] & data[3];
    let n3 = n1 ^ n2;
    return n3;
}

fn u8_to_binary_array(value: u8) -> [u8; 4] {
    let mut binary = [0u8; 4];
    for i in (0..4).rev() {
        binary[i] = (value >> i) & 0x1;
    }
    return binary;
}

fn main() {
    println!("Binary Machine Learning Algorithm");

    let max_steps = 100;
    let mut bnn = BinaryNN::new();

    /* 
    // all possible permutations for the function inputs
    // note: this is not yet the complete set, it would be every possibility - every number from 0 -> 255
    let function_inputs = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],

        [1, 1, 1, 1],
        [0, 0, 0, 0],

        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ];*/

    let function_inputs = (0..=u8::MAX).into_iter().map(|value: u8| -> [u8; 4] {
        let r = u8_to_binary_array(value);
        return r;
    }).collect::<Vec<_>>();

    // run the function over the inputs to generate a tuple of (input, output) pairs
    let mut training_data = function_inputs.iter().map(|inputs| {
        return (inputs, [example_function_to_learn(inputs)]);
    }).collect::<Vec<_>>();
    training_data.shuffle(&mut thread_rng());

    // learn on training data
    // we will likely need to loop over the complete traning data a few times in order to ensure we have learnt the function
    // so we continue until there is a pass where there are no more errors (or till we reach max_training_data_passes).
    let mut total_error_count = 0;
    let mut total_pass_error_count = 0;
    let max_training_data_passes = 1000;
    for _ in 0..max_training_data_passes {
        let mut errors_this_pass = 0;
        for (inputs, outputs) in training_data.iter() {
            let (_, error_count) = bnn.train(*inputs, outputs, max_steps);
            if error_count >= max_steps {
                println!("Failed to learn in the given steps?");
            }

            errors_this_pass += error_count;
            total_error_count += error_count;
        }

        if errors_this_pass == 0 {
            break;
        }

        total_pass_error_count += 1;
    }

    println!("");
    if total_pass_error_count == max_training_data_passes {
        println!("Failed to learn the algorithm with total error count: {} and total passes of: {}", total_error_count, total_pass_error_count);
    } else {
        println!("Learning is complete. Total training steps to learn the function: {}, with total passes of: {}", total_error_count, total_pass_error_count);
    }
    println!("");

    /*
    let in_1 = [1, 0, 0, 0];
    let out_1 = [example_function_to_learn(&in_1)];
    let train_result = bnn.train(&in_1, &out_1, max_steps);

    match train_result {
        TrainResult::Failure => println!("Failed to learn!"),
        TrainResult::Success => println!("Succesfully Learned!"),
    }
*/
    // verify that there are now no errors
    let mut final_train_result = TrainResult::Success;
    for (inputs, outputs) in training_data.iter() {
        bnn.forward_propagate(*inputs);
        let train_result = bnn.check_for_error(outputs);
        match train_result {
            TrainResult::Failure => {
                final_train_result = train_result;
                break;
            },
            TrainResult::Success => (),
        }
    }

    println!("");
    match final_train_result {
        TrainResult::Failure => println!("Failed to learn the algorithm."),
        TrainResult::Success => println!("Successfully learned the algorithm."),
    }
    println!("");
}
