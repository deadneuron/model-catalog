use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::Wgpu;
use burn::tensor::{
    Tensor,
    Distribution::Normal
};
use zoo::models::alexnet::{
    AlexNet, AlexNetConfig
};

fn main() {
    type B = Wgpu<AutoGraphicsApi, f32, i32>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    // Batches x Channels x Width x Height
    let samples: Tensor<B, 4> = Tensor::random([8, 3, 224, 224], Normal(0., 1.), &device);
        
    let model: AlexNet<B> = AlexNetConfig::new().init(&device);
    
    let outputs: Tensor<B, 2> = model.forward(samples);

    println!("{}", outputs);
    println!("{}", outputs.argmax(1))
}
