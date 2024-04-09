use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Linear, LinearConfig, ReLU,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct LeNet<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool1: AdaptiveAvgPool2d,
    pool2: AdaptiveAvgPool2d,
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    activation: ReLU,
}

#[derive(Config, Debug)]
pub struct LeNetConfig {
    #[config(default = "10")]
    num_classes: usize,
}

impl LeNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LeNet<B> {
        LeNet {
            conv1: Conv2dConfig::new([1, 6], [5, 5]).init(device),
            conv2: Conv2dConfig::new([6, 16], [5, 5],).init(device),
            pool1: AdaptiveAvgPool2dConfig::new([14, 14]).init(),
            pool2: AdaptiveAvgPool2dConfig::new([5, 5]).init(),
            linear1: LinearConfig::new(16*5*5, 120).init(device),
            linear2: LinearConfig::new(120, 84).init(device),
            linear3: LinearConfig::new(84, self.num_classes).init(device),
            activation: ReLU::new(),
        }
    }
}

impl<B: Backend> LeNet<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool1.forward(x);

        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool2.forward(x);

        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);

        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);

        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);

        self.linear3.forward(x)
    }
}
