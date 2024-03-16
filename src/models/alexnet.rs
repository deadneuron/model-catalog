use burn::{
  config::Config,
  module::Module,
  nn::{
    conv::{Conv2d, Conv2dConfig},
    PaddingConfig2d,
    pool::{
      MaxPool2d, MaxPool2dConfig, 
      AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig
    },
    Linear, LinearConfig, 
    Dropout, DropoutConfig,
    ReLU,
  },
  tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct AlexNet<B: Backend> {
  conv1: Conv2d<B>,
  conv2: Conv2d<B>,
  conv3: Conv2d<B>,
  conv4: Conv2d<B>,
  conv5: Conv2d<B>,
  maxpool: MaxPool2d,
  avgpool: AdaptiveAvgPool2d,
  linear1: Linear<B>,
  linear2: Linear<B>,
  linear3: Linear<B>,
  dropout: Dropout,
  activation: ReLU
}

#[derive(Config, Debug)]
pub struct AlexNetConfig {
  #[config(default = "1000")]
  num_classes: usize,
  #[config(default = "0.5")]
  dropout:f64,
}

impl AlexNetConfig {
  pub fn init<B: Backend>(&self, device: &B::Device) -> AlexNet<B> {
    AlexNet {
      conv1: Conv2dConfig::new([3, 64], [11, 11])
        .with_stride([4, 4])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      conv2: Conv2dConfig::new([64, 192], [5, 5])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      conv3: Conv2dConfig::new([192, 384], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      conv4: Conv2dConfig::new([384, 256], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      conv5: Conv2dConfig::new([256, 256], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      maxpool: MaxPool2dConfig::new([3, 3])
        .with_strides([2, 2])
        .init(),
      avgpool: AdaptiveAvgPool2dConfig::new([6, 6]).init(),
      linear1: LinearConfig::new(256*6*6, 4096).init(device),
      linear2: LinearConfig::new(4096, 4096).init(device),
      linear3: LinearConfig::new(4096, self.num_classes).init(device),
      dropout: DropoutConfig::new(self.dropout).init(),
      activation: ReLU::new(),
    }
  }
}

impl<B: Backend> AlexNet<B> {
  pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
    let x = self.conv1.forward(x);
    let x = self.activation.forward(x);
    let x = self.maxpool.forward(x);

    let x = self.conv2.forward(x);
    let x = self.activation.forward(x);
    let x = self.maxpool.forward(x);

    let x = self.conv3.forward(x);
    let x = self.activation.forward(x);

    let x = self.conv4.forward(x);
    let x = self.activation.forward(x);

    let x = self.conv5.forward(x);
    let x = self.activation.forward(x);
    let x = self.maxpool.forward(x);

    let x = self.avgpool.forward(x);

    let [batch_size, channels, height, width] = x.dims();
    let x = x.reshape([batch_size, channels * height * width]);
    
    let x = self.dropout.forward(x);
    let x = self.linear1.forward(x);
    let x = self.activation.forward(x);

    let x = self.dropout.forward(x);
    let x = self.linear2.forward(x);
    let x = self.activation.forward(x);

    self.linear3.forward(x)
  }
}