use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::Wgpu;
use burn::tensor::{
    Tensor,
    Distribution::Normal
};
use clap::{Parser, Subcommand, ValueEnum};
use model_catalog::models::{
    lenet::{LeNet, LeNetConfig},
    alexnet::{AlexNet, AlexNetConfig},
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
	/// [train, infer]
	#[command(subcommand)]
	task: Tasks,
}

#[derive(Subcommand)]
enum Tasks {
    Train {
    },
    Infer {
    }
}

fn main() {
    let args = Args::parse();

    type B = Wgpu<AutoGraphicsApi, f32, i32>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let model: LeNet<B> = LeNetConfig::new().init(&device);
    // let model: AlexNet<B> = AlexNetConfig::new().init(&device);


    match args.task {
        Tasks::Train { .. } => {},
        Tasks::Infer { .. } => {
            let samples: Tensor<B, 4> = Tensor::random([8, 1, 28, 28], Normal(0., 1.), &device);

            let outputs: Tensor<B, 2> = model.forward(samples);

            println!("{}", outputs);
            println!("{}", outputs.argmax(1))                
        }
    }
}