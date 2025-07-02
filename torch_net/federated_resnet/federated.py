
"""
Federated Learning Training Script for Comparing Real and Complex ResNets

This script uses Flower to simulate a federated learning environment for the
ResNet comparison experiment.

It defines the Flower Client, the server-side Strategy, and the main
simulation loop. All experiment parameters are controllable via CLI arguments.

Usage:
    # To run a default federated experiment with 10 clients for 5 rounds:
    python federated.py --num_clients 10 --num_rounds 5

    # To run only the 'WS' complex model with crelu:
    python federated.py --num_clients 10 --architectures WS --activations crelu --learn_imag_mode true_only
"""
import flwr as fl
import torch
import argparse
from itertools import product
from datetime import datetime
import logging
import os
import csv

from federated_utils import get_model, load_partitioned_data, train, test, get_weights, set_weights

# --- FLOWER CLIENT DEFINITION ---

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, valloader, device):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_weights(self.model)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_weights(self.model, parameters)
        
        epochs = config["local_epochs"]
        learning_rate = config["learning_rate"]
        
        train_loss, train_acc = train(self.model, self.trainloader, epochs=epochs, device=self.device, learning_rate=learning_rate)
        
        return get_weights(self.model), len(self.trainloader.dataset), {"train_loss": train_loss, "train_accuracy": train_acc}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_weights(self.model, parameters)
        
        loss, accuracy = test(self.model, self.valloader, device=self.device)
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

# --- SIMULATION SETUP ---

def client_fn(cid: str, config: dict, trainloaders, valloaders, device):
    """Create a Flower client instance for a given client ID."""
    model = get_model(config).to(device)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, model, trainloader, valloader, device)

def get_evaluate_fn(test_loader, device, config):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config_dict: dict):
        model = get_model(config)
        set_weights(model, parameters)
        model.to(device)
        loss, accuracy = test(model, test_loader, device)
        print(f"Server-side evaluation round {server_round} | Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")
        return loss, {"accuracy": accuracy}
    return evaluate

def save_results_to_csv(results, filename="federated_results.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print(f"Results saved to {filename}")

# --- MAIN DRIVER ---

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Starting federated training on device: {device}")

    trainloaders, valloaders, testloader = load_partitioned_data(args.num_clients, args.batch_size)

    arch_types = args.architectures
    complex_activations = args.activations
    if args.learn_imag_mode == 'true_only':
        learn_imag_opts = [True]
    elif args.learn_imag_mode == 'false_only':
        learn_imag_opts = [False]
    else:
        learn_imag_opts = [True, False]
        
    experiment_configs = []
    for arch, act, learn in product(arch_types, complex_activations, learn_imag_opts):
        experiment_configs.append({'name': f"Complex_ResNet_{arch}_{act}_{'LearnImag' if learn else 'ZeroImag'}", 'model_type': 'Complex', 'arch': arch, 'activation': act, 'learn_imag': learn})
    for arch in arch_types:
        experiment_configs.append({'name': f"Real_ResNet_{arch}", 'model_type': 'Real', 'arch': arch, 'activation': 'relu', 'learn_imag': 'N/A'})

    for config in experiment_configs:
        start_time = datetime.now()
        print("\n" + "="*80)
        print(f"STARTING NEW FEDERATED EXPERIMENT: {config['name']}")
        print(f"  - Architecture: {config['arch']}")
        print(f"  - Activation: {config['activation']}")
        print(f"  - Learn Imaginary: {config['learn_imag']}")
        print(f"  - Rounds: {args.num_rounds}")
        print(f"  - Clients: {args.num_clients}")
        print("="*80)

        bound_client_fn = lambda cid: client_fn(cid, config, trainloaders, valloaders, device)
        evaluate_fn = get_evaluate_fn(testloader, device, config)

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=args.fraction_fit,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=lambda server_round: {"local_epochs": args.local_epochs, "learning_rate": args.learning_rate},
        )

        history = fl.simulation.start_simulation(
            client_fn=bound_client_fn,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
            client_resources={"num_gpus": 1} if device.type == "cuda" else None,
        )

        final_accuracy = history.metrics_centralized["accuracy"][-1][1]
        final_loss = history.losses_centralized[-1][1]
        
        final_save_data = {
            'status': 'Completed',
            **config,
            'num_clients': args.num_clients,
            'num_rounds': args.num_rounds,
            'local_epochs': args.local_epochs,
            'final_server_accuracy': final_accuracy,
            'final_server_loss': final_loss,
            'training_time_seconds': (datetime.now() - start_time).total_seconds()
        }
        save_results_to_csv(final_save_data)
        print(f"SUCCESS: Experiment {config['name']} fully completed. Final Accuracy: {final_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Federated Learning for ResNet Comparison")
    
    parser.add_argument("--num_clients", type=int, default=10, help="Total number of clients to simulate.")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of federated learning rounds.")
    parser.add_argument("--local_epochs", type=int, default=5, help="Number of local epochs for each client.")
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fraction of clients to use for training.")
    parser.add_argument("--min_fit_clients", type=int, default=2, help="Minimum number of clients to train on.")
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for client training.')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for client training.')
    parser.add_argument('--architectures', '--arch', nargs='+', default=['WS', 'DN', 'IB'], choices=['WS', 'DN', 'IB'], help="Architectures to train.")
    parser.add_argument('--activations', '--act', nargs='+', default=['crelu', 'zrelu', 'modrelu', 'complex_cardioid'], choices=['crelu', 'zrelu', 'modrelu', 'complex_cardioid'], help="Complex activation functions to test.")
    parser.add_argument('--learn_imag_mode', default='both', choices=['true_only', 'false_only', 'both'], help="Controls which 'learn_imaginary_component' settings to use.")
    
    args = parser.parse_args()
    main(args)
