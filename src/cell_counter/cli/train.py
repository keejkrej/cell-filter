import argparse
from ..core.train import train_model

def main():
    parser = argparse.ArgumentParser(description='Train a cell counting model')
    parser.add_argument('annotations_path', type=str, help='Path to the annotations JSON file')
    parser.add_argument('images_dir', type=str, help='Directory containing the images')
    parser.add_argument('output_dir', type=str, help='Directory to save the model and results')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--image-size', type=int, nargs=2, default=[64, 64], help='Size to resize images to (height width)')
    parser.add_argument('--model-size', type=int, nargs=2, default=[224, 224], help='Size to resize images to for model input')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='L2 regularization strength')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate for regularization')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--confidence-threshold', type=float, default=0.9, help='Target confidence threshold for predictions')
    
    args = parser.parse_args()
    
    train_model(
        annotations_path=args.annotations_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        image_size=tuple(args.image_size),
        model_size=tuple(args.model_size),
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        max_grad_norm=args.max_grad_norm,
        confidence_threshold=args.confidence_threshold
    )

if __name__ == '__main__':
    main() 