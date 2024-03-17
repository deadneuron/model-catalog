use clap::{Parser, Subcommand};

/// Simple program to greet a person
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
	/// [classification]
	#[command(subcommand)]
	task: Tasks,
}

#[derive(Subcommand)]
enum Tasks {
	Classification {
		/// [training, inference]
		#[command(subcommand)]
		phase: Phases,
	}
}

#[derive(Subcommand)]
enum Phases {
	Training {
		/// [mnist, cifar]
		#[arg(short, long)]
		dataset: String,
		
		/// [lenet, alexnet]
		#[arg(short, long)]
		model: String,

		#[arg(long, default_value_t = 10)]
		epochs: usize,

		#[arg(long, default_value_t = 128)]
		batch_size: usize,

		/// [sgd, adam]
		#[arg(short, long, default_value_t = String::from("sgd"))]
		optimizer: String,
		
		#[arg(long, default_value_t = 0.01)]
		lr: f32,

		#[arg(long, default_value_t = 0.0005)]
		wd: f32,

		#[arg(long, default_value_t = 0.9)]
		momentum: f32,

		/// [constant, cosine]
		#[arg(long, default_value_t = String::from("cosine"))]
		schedule: String,
	},
	Inference {
		/// [mnist, cifar]
		#[arg(short, long)]
		dataset: String,
		
		/// [lenet, alexnet]
		#[arg(short, long)]
		model: String,

		/// Number of samples per batch
		#[arg(short, long, default_value_t = 100)]
		batch_size: usize,
	}
}

fn main() {
    let _args = Args::parse();
}