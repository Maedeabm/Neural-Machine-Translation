import subprocess

# Define the command and parameters as a list
command = [
    'python', 'train_nmt.py',
    '--attn_model', 'general',
    '--embedding_size', '256',
    '--hidden_size', '256',
    '--n_layers', '2',
    '--dropout', '0.1',
    '--teacher_forcing_ratio', '0.8',
    '--clip', '5.0',
    '--lr', '0.0005',
    '--n_epochs', '50000',
    '--plot_every', '200',
    '--print_every', '1000',
    '--language', 'spa',
    '--device', 'cuda',
    '--seed', '19'
]

# Run the command
subprocess.run(command)
