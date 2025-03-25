from trainer import SegmentationTrainer

if __name__ == "__main__":
    config = {
        "model_name": "segmenter_three",
        "data_path": "",
        "save_path": "models",
        "latent_dim": 256,
        "batch_size": 64,
        "num_epochs": 50,
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
        "random_seed": 42,
        "initial_lr": 0.0001,
        "decay_steps": 10000,
        "decay_rate": 0.9,
        "dropout_rate": 0.5,
        "patience": 5,
        "validation_split": 0.1
    }
    
    import json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

    trainer = SegmentationTrainer('config.json')
    model, data_processor = trainer.train()