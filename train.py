#!/usr/bin/env python3
"""
Enhanced Training Script for Admin Training Module
Integrates with the web interface for real-time monitoring
"""

# Force unbuffered output from the very start
import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

print("=" * 60, flush=True)
print("SCRIPT STARTING", flush=True)
print("=" * 60, flush=True)
print(f"Python: {sys.executable}", flush=True)
print(f"Working dir: {os.getcwd()}", flush=True)
sys.stdout.flush()

import json
import time
import logging
import argparse
# pymysql is optional - we use PHP API gateway instead
try:
    import pymysql
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False
    print("[INFO] pymysql not available - using PHP API gateway for database access", flush=True)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import re
import yaml

print("[OK] All imports successful", flush=True)
sys.stdout.flush()
# Optional imports - make training work even if these aren't installed
try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available - metrics will be limited")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available - plots will be skipped")

# ============================================================================
# LOAD FROM config.php (like Flask app does)
# ============================================================================

def load_config_from_php():
    """Load configuration from config.php file (same as Flask config.py)"""
    config = {}
    # config.php is in parent directory (Proto1/)
    config_php_path = Path(__file__).resolve().parent.parent / 'config.php'
    
    if config_php_path.exists():
        try:
            with open(config_php_path, 'r') as f:
                content = f.read()
            
            # Extract DB_HOST
            match = re.search(r"define\s*\(\s*['\"]DB_HOST['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_host'] = match.group(1)
            
            # Extract DB_USER
            match = re.search(r"define\s*\(\s*['\"]DB_USER['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_user'] = match.group(1)
            
            # Extract DB_PASS
            match = re.search(r"define\s*\(\s*['\"]DB_PASS['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_password'] = match.group(1)
            
            # Extract DB_NAME
            match = re.search(r"define\s*\(\s*['\"]DB_NAME['\"]\s*,\s*['\"]([^'\"]+)['\"]", content)
            if match:
                config['db_name'] = match.group(1)
            
            print(f"[OK] Loaded config from config.php: {config.get('db_host')} / {config.get('db_name')}", flush=True)
        except Exception as e:
            print(f"Warning: Could not read config.php: {e}", flush=True)
    else:
        print(f"Warning: config.php not found at {config_php_path}", flush=True)
    
    return config

# Load from config.php
php_config = load_config_from_php()

# Database configuration
# Priority: Environment variables > local defaults > config.php
# Use local database by default for training (online DB may not be accessible)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),  # Default to local root user
    'password': os.getenv('DB_PASSWORD', ''),  # Default to empty (local XAMPP)
    'database': os.getenv('DB_NAME', 'asdb'),  # Default to local database
    'charset': os.getenv('DB_CHARSET', 'utf8mb4')
}

# Database config - only needed if pymysql is available and we're running locally
# On Heroku, we use PHP API gateway instead
if PYMYSQL_AVAILABLE and not os.getenv('DB_HOST') and not os.getenv('DB_USER'):
    # Check if we can connect to local database first (local development only)
    try:
        test_conn = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='asdb'
        )
        test_conn.close()
        print("[OK] Using local database (localhost/root/asdb)", flush=True)
        DB_CONFIG = {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'asdb',
            'charset': 'utf8mb4'
        }
    except Exception:
        # If local fails, try config.php credentials
        print("[WARNING] Local database not accessible, trying config.php credentials...", flush=True)
        DB_CONFIG = {
            'host': php_config.get('db_host', 'localhost'),
            'user': php_config.get('db_user', 'root'),
            'password': php_config.get('db_password', ''),
            'database': php_config.get('db_name', 'asdb'),
            'charset': 'utf8mb4'
        }
        print(f"[INFO] Using database from config.php: {DB_CONFIG['host']} / {DB_CONFIG['database']}", flush=True)
else:
    # On Heroku or when pymysql is not available, use PHP API gateway
    DB_CONFIG = None
    print("[INFO] Using PHP API gateway for database access (no direct MySQL connection)", flush=True)

class AdminTrainingLogger:
    """Custom logger that writes to database"""
    
    def __init__(self, job_id, db_config):
        self.job_id = job_id
        self.db_config = db_config
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logging configuration - XAMPP compatible with Windows permission handling"""
        import sys
        import os
        
        # Logs go to parent directory (Proto1/training_logs/)
        script_dir = Path(__file__).resolve().parent
        log_dir = script_dir / "training_logs"
        
        # Create directory with proper error handling
        try:
            log_dir.mkdir(exist_ok=True)
        except PermissionError:
            # Use ASCII-safe message for Windows
            print(f"[WARNING] Cannot create training_logs directory. Using fallback location.", flush=True)
            # Fallback to temp directory or current directory
            log_dir = Path.cwd() / "training_logs"
            try:
                log_dir.mkdir(exist_ok=True)
            except Exception as e:
                # Use ASCII-safe message for Windows
                print(f"[WARNING] Cannot create fallback log directory: {e}", flush=True)
                # Last resort: use current directory
                log_dir = Path.cwd()
        
        # Windows: Try to set permissions (may not work on Windows)
        try:
            if os.name != 'nt':  # Not Windows
                os.chmod(str(log_dir), 0o777)
        except Exception:
            pass  # Ignore permission errors on Windows
        
        # Force unbuffered output
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(line_buffering=True)
        
        log_file_path = log_dir / f"job_{self.job_id}.log"
        
        # Try to create log file with error handling
        handlers = [logging.StreamHandler(sys.stdout)]  # Always include console
        
        try:
            # Test if we can write to the log file by actually trying to open it
            test_handle = None
            try:
                test_handle = open(log_file_path, 'a', encoding='utf-8')
                test_handle.write("")  # Try to write
                test_handle.close()
                
                # If successful, add file handler
                handlers.append(logging.FileHandler(log_file_path, mode='a', encoding='utf-8'))
                # Use ASCII-safe message for Windows
                print(f"[OK] Logging to file: {log_file_path.absolute()}", flush=True)
            except (PermissionError, OSError) as e:
                if test_handle:
                    test_handle.close()
                raise  # Re-raise to outer except
        except (PermissionError, OSError) as e:
            # Use ASCII-safe messages for Windows compatibility
            print(f"[WARNING] Cannot write to log file {log_file_path}: {e}", flush=True)
            print(f"[WARNING] Logging to console only. Check directory permissions.", flush=True)
            # Continue with console-only logging
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True
        )
        self.logger = logging.getLogger(__name__)
        
        # Immediately write to log file to confirm it works (if file handler exists)
        try:
            if len(handlers) > 1:  # File handler was added
                self.logger.info(f"Log file created: {log_file_path.absolute()}")
            else:
                self.logger.info("Logging to console only (file logging unavailable)")
        except Exception:
            # If even logging fails, just continue silently
            pass
        sys.stdout.flush()
    
    def info(self, message):
        """Log info message"""
        # Ensure message is ASCII-safe for Windows
        safe_message = str(message).encode('ascii', 'replace').decode('ascii')
        self.logger.info(safe_message)
        self.log_to_db('INFO', safe_message)
    
    def warning(self, message):
        """Log warning message"""
        # Ensure message is ASCII-safe for Windows
        safe_message = str(message).encode('ascii', 'replace').decode('ascii')
        self.logger.warning(safe_message)
        self.log_to_db('WARNING', safe_message)
    
    def error(self, message):
        """Log error message"""
        # Ensure message is ASCII-safe for Windows
        safe_message = str(message).encode('ascii', 'replace').decode('ascii')
        self.logger.error(safe_message)
        self.log_to_db('ERROR', safe_message)
    
    def log_to_db(self, level, message):
        """Log message to database via PHP API (optional - won't crash if API unavailable)"""
        try:
            # Ensure message is ASCII-safe
            safe_message = str(message).encode('ascii', 'replace').decode('ascii')
            
            # Use PHP API instead of direct database connection
            import requests
            php_api_base = os.getenv('PHP_API_BASE', 'https://agrishield.bccbsis.com/api/training')
            url = f"{php_api_base}/add_log.php"
            
            data = {
                'job_id': self.job_id,
                'level': level,
                'message': safe_message[:1000]  # Limit length
            }
            
            # Non-blocking: don't wait too long for API response
            requests.post(url, json=data, timeout=2)
        except Exception:
            # Silently fail - API logging is optional
            # Training continues even if logging fails
            pass

class EnhancedPestDataset(Dataset):
    """Enhanced dataset with better error handling and statistics"""
    
    def __init__(self, data_dir, transform=None, logger=None, classes_from_yaml=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.logger = logger
        self.samples = []
        self.class_counts = {}
        self.classes_from_yaml = classes_from_yaml  # Classes from data.yaml
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset with statistics"""
        if not self.data_dir.exists():
            if self.logger:
                self.logger.error(f"Dataset directory not found: {self.data_dir}")
            return
        
        # PRIORITY: Use classes from YAML if provided, otherwise detect from directories
        if self.classes_from_yaml:
            self.classes = self.classes_from_yaml.copy()
            self.classes.sort()
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            if self.logger:
                self.logger.info(f"Using classes from data.yaml: {self.classes}")
        else:
            # Fallback: Get pest classes from directory structure
            self.classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
            self.classes.sort()
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            if self.logger:
                self.logger.info(f"Found classes from directory structure: {self.classes}")
        
        # Collect all image paths and labels
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_images = []
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    label = self.class_to_idx[class_name]
                    self.samples.append((str(img_path), label))
                    class_images.append(img_path)
            
            self.class_counts[class_name] = len(class_images)
            
            if self.logger:
                self.logger.info(f"{class_name}: {len(class_images)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_statistics(self):
        """Get dataset statistics"""
        total_images = len(self.samples)
        return {
            'total_images': total_images,
            'class_counts': self.class_counts,
            'classes': self.classes
        }

class ModelTrainer:
    """Enhanced model trainer with database integration"""
    
    def __init__(self, job_id, config, logger):
        self.job_id = job_id
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Railway is CPU-only (no GPU instances). Use all available cores.
        if self.device.type == 'cpu':
            threads = max(1, min(8, (os.cpu_count() or 2)))
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(max(1, threads // 2))
        self.logger.info(
            f"Compute device: {self.device}"
            + (f" ({torch.cuda.get_device_name(0)})" if self.device.type == 'cuda' else " [Railway has no GPU — CPU training]")
        )
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def create_model(self, num_classes):
        """Create ResNet18 model for classification"""
        self.logger.info(f"Creating ResNet18 model with {num_classes} classes")
        
        # Load pre-trained ResNet18
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the final layer for our number of classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        return model
    
    def get_data_transforms(self):
        """Get data transforms for training and validation"""
        
        # Training transforms — keep light on CPU (Railway has no GPU)
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms (no augmentation)
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transforms, val_transforms
    
    def train_epoch(self, model, dataloader, criterion, optimizer):
        """Train for one epoch - YOLOv8 style output"""
        import sys
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # YOLOv8-style progress output (like CMD training)
            batch_progress = (batch_idx + 1) / len(dataloader) * 100
            current_acc = 100. * correct / total if total > 0 else 0
            
            # Print progress every batch (like YOLOv8)
            # Use ASCII-safe characters for Windows compatibility
            progress_bar_length = 30
            filled = int(progress_bar_length * batch_progress / 100)
            bar = '#' * filled + '-' * (progress_bar_length - filled)  # ASCII-safe progress bar
            
            # Format like YOLOv8: epoch/batch  loss  accuracy  progress_bar
            progress_line = f"  {batch_idx+1}/{len(dataloader)}  {loss.item():.4f}  {current_acc:.1f}%  [{bar}] {batch_progress:.0f}%"
            print(progress_line, end='\r', flush=True)
            
            # Log progress less often to PHP (HTTP logging was slowing CPU runs)
            if batch_idx % 20 == 0 or batch_idx == len(dataloader) - 1:
                print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%", flush=True)
                self.logger.info(f'Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%')
                sys.stdout.flush()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, dataloader, criterion):
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_metrics_to_db(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Save training metrics to database (optional - only if pymysql available)"""
        if not PYMYSQL_AVAILABLE or not DB_CONFIG:
            # Skip if pymysql not available (we're on Heroku using PHP API)
            return
        try:
            conn = pymysql.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO training_metrics (training_job_id, epoch, accuracy, loss, val_accuracy, val_loss) VALUES (%s, %s, %s, %s, %s, %s)",
                (self.job_id, epoch, train_acc/100, train_loss, val_acc/100, val_loss)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to save metrics to database: {e}")
    
    def train(self, train_dataset, val_dataset):
        """Main training function"""
        self.logger.info("Starting training process")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == 'cuda'),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == 'cuda'),
        )
        
        # Create model
        num_classes = len(train_dataset.classes)
        self.logger.info(f"Dataset contains {num_classes} classes: {train_dataset.classes}")
        print(f"Number of classes detected: {num_classes}", flush=True)
        print(f"Classes: {train_dataset.classes}", flush=True)
        model = self.create_model(num_classes)
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
        # Training loop - YOLOv8 style output
        import sys
        print("\n" + "="*60, flush=True)
        print("TRAINING STARTED", flush=True)
        print("="*60, flush=True)
        print(f"Epochs: {self.config['epochs']}", flush=True)
        print(f"Batch Size: {self.config['batch_size']}", flush=True)
        print(f"Learning Rate: {self.config['learning_rate']}", flush=True)
        print(f"Train Batches: {len(train_loader)}", flush=True)
        print(f"Val Batches: {len(val_loader)}", flush=True)
        print("="*60 + "\n", flush=True)
        sys.stdout.flush()
        
        for epoch in range(self.config['epochs']):
            # YOLOv8-style epoch header
            print(f"\n{'='*60}", flush=True)
            print(f"Epoch {epoch+1}/{self.config['epochs']}", flush=True)
            print(f"{'='*60}", flush=True)
            sys.stdout.flush()
            
            self.logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            
            # Train with progress output
            print(f"\nTraining:", flush=True)
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            print()  # New line after progress bar
            
            # Validate
            print(f"\nValidating:", flush=True)
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # Save metrics
            self.save_metrics_to_db(epoch+1, train_loss, train_acc, val_loss, val_acc)
            
            # Store in history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # YOLOv8-style epoch summary
            print(f"\nEpoch {epoch+1} Summary:", flush=True)
            print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%", flush=True)
            print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%", flush=True)
            sys.stdout.flush()
            
            # Log epoch results
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model (locally only, will upload at end of training)
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model(model, val_acc, train_dataset.classes, upload=False)
                print(f"  [OK] New best model saved! (Accuracy: {val_acc:.2f}%)", flush=True)
                sys.stdout.flush()
        
        return model
    
    def convert_to_onnx(self, model, model_path, input_size=(3, 224, 224)):
        """Convert PyTorch model to ONNX format"""
        try:
            onnx_path = model_path.with_suffix('.onnx')
            
            self.logger.info(f"Converting model to ONNX format...")
            print(f"[INFO] Converting model to ONNX...", flush=True)
            
            # Set model to evaluation mode
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, *input_size)
            
            # Export to ONNX (torch.onnx.export is built into PyTorch)
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            # Validate ONNX model (optional)
            try:
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                self.logger.info(f"[OK] ONNX model validated successfully")
            except ImportError:
                # onnx package not available, skip validation
                pass
            except Exception as e:
                self.logger.warning(f"ONNX validation warning: {e}")
            
            onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"[OK] ONNX model saved to: {onnx_path} ({onnx_size_mb:.2f} MB)")
            print(f"[OK] ONNX model saved: {onnx_path.name} ({onnx_size_mb:.2f} MB)", flush=True)
            
            return onnx_path
            
        except Exception as e:
            self.logger.error(f"Failed to convert to ONNX: {e}")
            print(f"[ERROR] ONNX conversion failed: {e}", flush=True)
            return None
    
    def upload_model_to_server(self, model_path, accuracy, model_type='onnx'):
        """Upload model file to web server via PHP API with retry logic and SSL error handling"""
        import requests
        import base64
        import time
        
        # Try to import retry utilities (optional)
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            RETRY_AVAILABLE = True
        except ImportError:
            RETRY_AVAILABLE = False
            self.logger.warning("urllib3 retry not available, using simple retry logic")
        
        php_api_base = os.getenv('PHP_API_BASE', 'https://agrishield.bccbsis.com/api/training')
        upload_url = f"{php_api_base}/upload_model.php"
        
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return False
        
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Uploading model to server... (Size: {model_size_mb:.2f} MB)")
        print(f"[INFO] Uploading model to server... ({model_size_mb:.2f} MB)", flush=True)
        
        # Read model file - use multipart/form-data for better efficiency with large files
        try:
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
            # For large files, use multipart/form-data instead of base64 JSON
            # This is more memory efficient and faster
            self.logger.info(f"Model file read ({model_size_mb:.2f} MB)")
        except Exception as e:
            self.logger.error(f"Failed to read model file: {e}")
            return False
        
        # Get farm_id from training job (if available)
        farm_id = 0
        try:
            # Try to get farm_id from training_jobs table via PHP API
            php_api_base = os.getenv('PHP_API_BASE', 'https://agrishield.bccbsis.com/api/training')
            job_info_url = f"{php_api_base}/get_training_job_info.php"
            job_response = requests.get(f"{job_info_url}?job_id={self.job_id}", timeout=5)
            if job_response.status_code == 200:
                job_info = job_response.json()
                if job_info.get('success') and job_info.get('farm_parcels_id'):
                    farm_id = int(job_info.get('farm_parcels_id', 0))
                    if farm_id > 0:
                        self.logger.info(f"Found farm_id from training job: {farm_id}")
                        print(f"[INFO] Training for farm_id: {farm_id}", flush=True)
        except Exception as e:
            # Silently fail - farm_id will be retrieved by upload_model.php from database
            pass
        
        # Prepare upload data - use files parameter for multipart upload
        # This is more efficient than base64 encoding in JSON
        upload_data = {
            'job_id': str(self.job_id),
            'accuracy': str(accuracy),
            'model_type': model_type,
            'model_size_mb': str(model_size_mb)
        }
        
        # Add farm_id if available
        if farm_id > 0:
            upload_data['farm_id'] = str(farm_id)
        
        # Prepare file for multipart upload
        files = {
            'model_file': (model_path.name, model_bytes, 'application/octet-stream')
        }
        
        # Create session WITHOUT automatic retry on 500 errors (we handle manually)
        # This allows us to see the actual error message
        if RETRY_AVAILABLE:
            # Only retry on connection errors, not on 500 errors (so we can see the error)
            retry_strategy = Retry(
                total=2,  # Only 2 retries for connection issues
                backoff_factor=1,
                status_forcelist=[429, 502, 503, 504],  # Don't auto-retry 500, we handle it manually
                allowed_methods=["POST"]
            )
            session = requests.Session()
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        else:
            session = requests.Session()
        
        # Upload with retries and better error handling
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                self.logger.info(f"Upload attempt {attempt}/{max_attempts}...")
                print(f"[INFO] Upload attempt {attempt}/{max_attempts}...", flush=True)
                
                # Use multipart/form-data for large files (more efficient than JSON base64)
                # Use longer timeout for large files (10 minutes)
                response = session.post(
                    upload_url,
                    data=upload_data,
                    files=files,
                    timeout=600,  # 10 min timeout for large models
                    verify=True,  # Verify SSL certificate
                    stream=False  # Don't stream, send all at once
                )
                
                # Log response details for debugging
                self.logger.info(f"Response status: {response.status_code}")
                self.logger.info(f"Response headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if result.get('success'):
                            model_info = result.get('model', {})
                            self.logger.info(f"[OK] Model uploaded successfully: {model_info.get('version', 'N/A')}")
                            self.logger.info(f"  Path: {model_info.get('path', 'N/A')}")
                            self.logger.info(f"  Size: {model_info.get('size_mb', 0):.2f} MB")
                            self.logger.info(f"  Accuracy: {accuracy:.2f}%")
                            print(f"[OK] Model uploaded and activated: {model_info.get('version', 'N/A')}", flush=True)
                            return True
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            error_details = result.get('mysql_error') or result.get('php_error') or result.get('error')
                            self.logger.error(f"Upload failed: {error_msg}")
                            if error_details and error_details != error_msg:
                                self.logger.error(f"Error details: {error_details}")
                            print(f"[ERROR] Upload failed: {error_msg}", flush=True)
                            if attempt < max_attempts:
                                wait_time = 2 ** attempt
                                print(f"[WARN] Retrying in {wait_time} seconds...", flush=True)
                                time.sleep(wait_time)
                                continue
                            return False
                    except ValueError as e:
                        # Not JSON response
                        error_text = response.text[:1000] if response.text else "No response body"
                        self.logger.error(f"Invalid JSON response: {error_text}")
                        print(f"[ERROR] Server returned invalid JSON: {error_text[:200]}", flush=True)
                        if attempt < max_attempts:
                            wait_time = 2 ** attempt
                            time.sleep(wait_time)
                            continue
                        return False
                else:
                    # HTTP error - try to parse JSON error response
                    error_text = response.text[:2000] if response.text else "No error message"
                    self.logger.error(f"Upload failed: HTTP {response.status_code}")
                    self.logger.error(f"Response body: {error_text}")
                    
                    # Try to parse as JSON to get detailed error
                    try:
                        error_json = response.json()
                        error_msg = error_json.get('error', 'Unknown error')
                        error_details = error_json.get('mysql_error') or error_json.get('php_error') or error_json.get('error')
                        self.logger.error(f"Server error: {error_msg}")
                        if error_details and error_details != error_msg:
                            self.logger.error(f"Error details: {error_details}")
                        print(f"[ERROR] HTTP {response.status_code}: {error_msg}", flush=True)
                    except:
                        print(f"[ERROR] HTTP {response.status_code}: {error_text[:200]}", flush=True)
                    
                    if attempt < max_attempts:
                        wait_time = 2 ** attempt
                        print(f"[WARN] Retrying in {wait_time} seconds...", flush=True)
                        time.sleep(wait_time)
                        continue
                    return False
                    
            except requests.exceptions.SSLError as e:
                self.logger.warning(f"SSL error on attempt {attempt}: {e}")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    print(f"[WARN] SSL error, retrying in {wait_time} seconds...", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"SSL error after {max_attempts} attempts: {e}")
                    print(f"[ERROR] SSL connection failed after {max_attempts} attempts", flush=True)
                    return False
                    
            except requests.exceptions.Timeout as e:
                self.logger.warning(f"Timeout on attempt {attempt}: {e}")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    print(f"[WARN] Timeout, retrying in {wait_time} seconds...", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"Timeout after {max_attempts} attempts")
                    print(f"[ERROR] Upload timeout after {max_attempts} attempts", flush=True)
                    return False
                    
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"Connection error on attempt {attempt}: {e}")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    print(f"[WARN] Connection error, retrying in {wait_time} seconds...", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"Connection error after {max_attempts} attempts: {e}")
                    print(f"[ERROR] Connection failed after {max_attempts} attempts", flush=True)
                    return False
                    
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt}: {e}")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    print(f"[WARN] Error occurred, retrying in {wait_time} seconds...", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"Failed to upload model after {max_attempts} attempts: {e}")
                    print(f"[ERROR] Model upload error: {e}", flush=True)
                    return False
        
        return False
    
    def save_model(self, model, accuracy, classes, upload=False):
        """Save model to database and file system - Only uploads if upload=True (at end of training)"""
        try:
            # Create model directory (in parent directory, same as root)
            script_dir = Path(__file__).resolve().parent
            model_dir = script_dir / "models" / f"job_{self.job_id}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save PyTorch model file
            model_path = model_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            
            # Get model size
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            self.logger.info(f"[OK] PyTorch model saved to: {model_path} ({model_size_mb:.2f} MB)")
            print(f"[OK] Model saved: {model_path.name} ({model_size_mb:.2f} MB)", flush=True)
            
            # Convert to ONNX for inference
            onnx_path = self.convert_to_onnx(model, model_path)
            
            # Only upload if explicitly requested (at end of training)
            upload_success = False
            if upload:
                if onnx_path and onnx_path.exists():
                    upload_success = self.upload_model_to_server(onnx_path, accuracy, 'onnx')
                else:
                    # Fallback: upload PyTorch model
                    self.logger.info("Uploading PyTorch model (ONNX conversion not available)")
                    upload_success = self.upload_model_to_server(model_path, accuracy, 'pth')
            else:
                self.logger.info("Model saved locally (will upload after all epochs complete)")
                print(f"[INFO] Model saved locally (upload after training)", flush=True)
            
            # Save to database - create or update entry for this training job
            # IMPROVED: Checks for existing model and only updates if accuracy is better
            if PYMYSQL_AVAILABLE and DB_CONFIG:
                try:
                    conn = pymysql.connect(**DB_CONFIG)
                    cursor = conn.cursor()
                    
                    # Check if model already exists for this job_id
                    cursor.execute(
                        "SELECT model_id FROM model_versions WHERE training_job_id = %s",
                        (self.job_id,)
                    )
                    existing_model = cursor.fetchone()
                    
                    # Get farm_id from training_jobs table (for farm-specific models)
                    farm_id = None
                    cursor.execute(
                        "SELECT farm_parcels_id FROM training_jobs WHERE job_id = %s",
                        (self.job_id,)
                    )
                    job_result = cursor.fetchone()
                    if job_result and job_result[0]:
                        farm_id = job_result[0]
                    
                    # Prepare classes JSON
                    classes_json = None
                    if classes:
                        classes_json = json.dumps(list(classes))
                    
                    # Use uploaded path if available, otherwise local path
                    model_db_path = onnx_path.name if onnx_path else model_path.name
                    accuracy_decimal = accuracy / 100.0
                    
                    if existing_model:
                        # Update existing entry (only keep best model per job)
                        model_id = existing_model[0]
                        
                        # Check if this model has better accuracy than existing
                        cursor.execute(
                            "SELECT accuracy FROM model_versions WHERE model_id = %s",
                            (model_id,)
                        )
                        existing_acc_result = cursor.fetchone()
                        existing_accuracy = existing_acc_result[0] if existing_acc_result else 0.0
                        
                        # Only update if this accuracy is better
                        if accuracy_decimal > existing_accuracy:
                            # Update model accuracy and size
                            cursor.execute(
                                "UPDATE model_versions SET accuracy = %s, model_size_mb = %s, deployed_at = NOW() WHERE model_id = %s",
                                (accuracy_decimal, model_size_mb, model_id)
                            )
                            
                            # Update classes_json if column exists
                            if classes_json:
                                try:
                                    cursor.execute(
                                        "UPDATE model_versions SET classes_json = %s WHERE model_id = %s",
                                        (classes_json, model_id)
                                    )
                                except Exception:
                                    # classes_json column might not exist - ignore
                                    pass
                            
                            self.logger.info(f"[OK] Model updated in database (Accuracy improved: {accuracy:.2f}%)")
                        else:
                            self.logger.info(f"Existing model has better accuracy ({existing_accuracy*100:.2f}% > {accuracy:.2f}%), not updating")
                    else:
                        # Create new entry
                        # Get next version number
                        cursor.execute(
                            "SELECT MAX(CAST(SUBSTRING(version, 2) AS UNSIGNED)) as max_version FROM model_versions"
                        )
                        version_result = cursor.fetchone()
                        max_version = version_result[0] if version_result and version_result[0] else 0
                        next_version = max_version + 1
                        version_str = f"v{next_version}"
                        
                        # Determine if farm-specific
                        is_farm_specific = (farm_id is not None and farm_id > 0)
                        is_current = 0 if is_farm_specific else 1
                        
                        # Insert new model - try with classes_json first
                        try:
                            if classes_json:
                                cursor.execute(
                                    "INSERT INTO model_versions (version, model_path, accuracy, training_job_id, model_size_mb, is_active, is_current, deployed_at, classes_json) VALUES (%s, %s, %s, %s, %s, 1, %s, NOW(), %s)",
                                    (version_str, model_db_path, accuracy_decimal, self.job_id, model_size_mb, is_current, classes_json)
                                )
                            else:
                                cursor.execute(
                                    "INSERT INTO model_versions (version, model_path, accuracy, training_job_id, model_size_mb, is_active, is_current, deployed_at) VALUES (%s, %s, %s, %s, %s, 1, %s, NOW())",
                                    (version_str, model_db_path, accuracy_decimal, self.job_id, model_size_mb, is_current)
                                )
                        except Exception as e:
                            # If classes_json column doesn't exist, try without it
                            if 'classes_json' in str(e):
                                cursor.execute(
                                    "INSERT INTO model_versions (version, model_path, accuracy, training_job_id, model_size_mb, is_active, is_current, deployed_at) VALUES (%s, %s, %s, %s, %s, 1, %s, NOW())",
                                    (version_str, model_db_path, accuracy_decimal, self.job_id, model_size_mb, is_current)
                                )
                            else:
                                raise
                        
                        model_id = cursor.lastrowid
                        
                        # If farm-specific, auto-assign to farm
                        if is_farm_specific:
                            try:
                                # Check if farm_model_assignments table exists
                                cursor.execute("SHOW TABLES LIKE 'farm_model_assignments'")
                                if cursor.fetchone():
                                    # Auto-assign model to farm
                                    cursor.execute(
                                        "INSERT INTO farm_model_assignments (farm_parcels_id, model_id, assigned_by) VALUES (%s, %s, 1) ON DUPLICATE KEY UPDATE model_id = VALUES(model_id), assigned_at = CURRENT_TIMESTAMP",
                                        (farm_id, model_id)
                                    )
                                    self.logger.info(f"Model automatically assigned to farm_id: {farm_id}")
                            except Exception as e:
                                self.logger.warning(f"Could not auto-assign model to farm: {e}")
                        
                        # If global model, deactivate previous global models
                        if not is_farm_specific:
                            try:
                                cursor.execute(
                                    "UPDATE model_versions SET is_active = 0, is_current = 0 WHERE is_current = 1 AND model_id != %s",
                                    (model_id,)
                                )
                            except Exception as e:
                                self.logger.warning(f"Could not deactivate previous models: {e}")
                        
                        self.logger.info(f"[OK] Model {version_str} created in database (Accuracy: {accuracy:.2f}%)")
                    
                    conn.commit()
                    conn.close()
                    
                    self.logger.info(f"[OK] Model saved locally AND in database (Accuracy: {accuracy:.2f}%, Size: {model_size_mb:.2f} MB)")
                    
                except Exception as db_error:
                    # Database save is optional - don't fail training if DB is unavailable
                    self.logger.warning(f"Could not save model to database: {db_error}")
                    self.logger.info(f"[OK] Model saved locally only (Accuracy: {accuracy:.2f}%, Size: {model_size_mb:.2f} MB)")
                    self.logger.info(f"[WARNING] Database entry not created - will be created when model is uploaded")
            
            if upload and upload_success:
                self.logger.info(f"[OK] Model uploaded to server and activated successfully!")
                print(f"[OK] Model is now active and ready for detection!", flush=True)
            elif upload and not upload_success:
                self.logger.warning(f"[WARNING] Model saved locally but upload to server failed")
                print(f"[WARNING] Model saved but not uploaded - check server connection", flush=True)
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            print(f"[ERROR] Failed to save model: {e}", flush=True)

def update_job_status(job_id, status, error_message=None):
    """Update training job status via PHP API"""
    try:
        import requests
        php_api_base = os.getenv('PHP_API_BASE', 'https://agrishield.bccbsis.com/api/training')
        url = f"{php_api_base}/update_status.php"
        
        data = {
            'job_id': job_id,
            'status': status
        }
        if error_message:
            data['message'] = str(error_message).encode('ascii', 'replace').decode('ascii')[:500]
        
        requests.post(url, json=data, timeout=5)
    except Exception:
        # Silently fail - API updates are optional
        pass

def load_classes_from_yaml(yaml_path):
    """Load class names from data.yaml file"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        # Try to get names from yaml
        if 'names' in yaml_data:
            names = yaml_data['names']
            # Handle both list and dict formats
            if isinstance(names, list):
                # Convert display names back to class codes (lowercase, replace spaces with underscores)
                class_codes = [name.lower().replace(' ', '_').replace('-', '_') for name in names]
                return class_codes
            elif isinstance(names, dict):
                # If it's a dict like {0: 'class1', 1: 'class2'}, get values
                class_codes = [name.lower().replace(' ', '_').replace('-', '_') for name in names.values()]
                return class_codes
        
        return None
    except Exception as e:
        print(f"Warning: Could not load classes from {yaml_path}: {e}", flush=True)
        return None

def download_dataset_from_server(script_dir, logger, job_id=None, farm_id=None, selected_pests=None):
    """Download organized dataset from DATASET_ZIP_URL (preferred) or PHP API."""
    try:
        import requests
        import zipfile
        import tempfile

        from dataset_source import (
            dataset_ready,
            download_external_dataset_zip,
            get_dataset_zip_url,
            normalize_dataset_layout,
        )

        script_dir = Path(script_dir)
        external_url = get_dataset_zip_url()

        # Option 2: external Drive/ZIP — must succeed when configured (no silent PHP fallback)
        try:
            if download_external_dataset_zip(script_dir, logger=logger):
                return True
        except RuntimeError as ext_err:
            print(f"[ERROR] {ext_err}", flush=True)
            logger.error(str(ext_err))
            return False

        if external_url:
            # URL set but returned False unexpectedly
            print("[ERROR] DATASET_ZIP_URL is set but dataset is not ready", flush=True)
            return False

        # Normalize any leftover extract from a previous run
        if normalize_dataset_layout(script_dir):
            print("[INFO] Using normalized local dataset", flush=True)
            return True

        # Check if dataset already exists locally
        organized_dir = script_dir / "training_data" / "dataset_organized"
        has_dataset = dataset_ready(script_dir)
        if has_dataset:
            print("[INFO] Dataset already exists locally, skipping download", flush=True)
            return True
        
        # Get PHP API base URL (only when no DATASET_ZIP_URL)
        php_api_base = os.getenv('PHP_API_BASE', 'https://agrishield.bccbsis.com/api/training')
        download_url = f"{php_api_base}/download_dataset.php?format=zip"
        
        # Add farm_id and selected_pests if provided
        if job_id:
            # Try to get farm_id from training job if not provided
            if not farm_id:
                try:
                    job_info_url = f"{php_api_base}/get_training_job_info.php"
                    job_response = requests.get(f"{job_info_url}?job_id={job_id}", timeout=5)
                    if job_response.status_code == 200:
                        job_info = job_response.json()
                        if job_info.get('success') and job_info.get('farm_parcels_id'):
                            farm_id = int(job_info.get('farm_parcels_id', 0))
                            if farm_id > 0:
                                print(f"[INFO] Found farm_id from training job: {farm_id}", flush=True)
                                logger.info(f"Found farm_id from training job: {farm_id}")
                                
                                # Get selected_pests from training_config if available
                                training_config = job_info.get('training_config', {})
                                if isinstance(training_config, dict) and training_config.get('selected_pests'):
                                    selected_pests = training_config.get('selected_pests')
                except Exception as e:
                    print(f"[WARN] Could not get farm_id from job: {e}", flush=True)
                    pass
        
        # Build URL with parameters
        url_params = []
        if farm_id and farm_id > 0:
            url_params.append(f"farm_id={farm_id}")
        if selected_pests and isinstance(selected_pests, list) and len(selected_pests) > 0:
            # URL encode the JSON string to avoid issues with special characters
            import urllib.parse
            pests_json = json.dumps(selected_pests)
            pests_encoded = urllib.parse.quote(pests_json)
            url_params.append(f"selected_pests={pests_encoded}")
        
        if url_params:
            download_url += "&" + "&".join(url_params)
            print(f"[INFO] Downloading farm-specific dataset (farm_id: {farm_id})", flush=True)
            logger.info(f"Downloading farm-specific dataset (farm_id: {farm_id})")
        
        print(f"[INFO] Downloading dataset from server...", flush=True)
        print(f"  URL: {download_url}", flush=True)
        logger.info(f"Downloading dataset from {download_url}")
        
        # Download ZIP file
        response = requests.get(download_url, timeout=300, stream=True)  # 5 min timeout for large files
        
        if response.status_code != 200:
            print(f"[ERROR] Failed to download dataset: HTTP {response.status_code}", flush=True)
            logger.error(f"Dataset download failed: HTTP {response.status_code}")
            if response.status_code == 404:
                print("[ERROR] Dataset not found on server. Please upload a dataset first.", flush=True)
                print("[ERROR] Go to admin training module and upload/organize a dataset.", flush=True)
            elif response.status_code == 500:
                print("[ERROR] Server error when downloading dataset. Check server logs.", flush=True)
            # Try to get error message from response
            try:
                error_data = response.json()
                if 'error' in error_data:
                    print(f"[ERROR] Server error: {error_data['error']}", flush=True)
            except:
                pass
            return False
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_zip_path = tmp_file.name
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
        
        zip_size = os.path.getsize(tmp_zip_path)
        print(f"[OK] Dataset downloaded ({zip_size / 1024 / 1024:.1f} MB)", flush=True)
        logger.info(f"Dataset downloaded: {zip_size / 1024 / 1024:.1f} MB")
        
        # Check if ZIP is too small (likely empty)
        if zip_size < 1024:  # Less than 1KB is suspicious
            print(f"[ERROR] Downloaded ZIP file is too small ({zip_size} bytes). Dataset may be empty on server.", flush=True)
            logger.error(f"Downloaded ZIP file is too small: {zip_size} bytes")
            os.unlink(tmp_zip_path)
            return False

        with open(tmp_zip_path, 'rb') as _zf:
            magic = _zf.read(4)
        if magic[:2] != b'PK':
            # Hostinger often returns JSON/HTML error pages with 200
            preview = ''
            try:
                with open(tmp_zip_path, 'rb') as _zf:
                    preview = _zf.read(200).decode('utf-8', errors='replace')
            except Exception:
                pass
            print("[ERROR] PHP download_dataset.php did not return a ZIP file.", flush=True)
            print(f"[ERROR] Preview: {preview[:180]}", flush=True)
            print("[ERROR] Set Railway DATASET_ZIP_URL to your Google Drive folder instead.", flush=True)
            logger.error("Downloaded file is not a zip file")
            os.unlink(tmp_zip_path)
            return False
        
        # Extract ZIP
        print("[INFO] Extracting dataset...", flush=True)
        logger.info("Extracting dataset")
        
        # Create organized_dir directly (where we want the dataset)
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        # Check ZIP structure first
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Check if ZIP has files
            if not file_list:
                print("[ERROR] ZIP file is empty. Dataset on server has no files.", flush=True)
                logger.error("ZIP file is empty")
                os.unlink(tmp_zip_path)
                return False
            
            # Count image files in ZIP
            image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"[INFO] ZIP contains {len(file_list)} total files, {len(image_files)} image files", flush=True)
            if len(image_files) == 0:
                print("[ERROR] ZIP file contains no image files. Dataset on server is empty or has wrong structure.", flush=True)
                logger.error("ZIP file contains no image files")
                os.unlink(tmp_zip_path)
                return False
            
            # Show sample of ZIP structure for debugging
            print(f"[DEBUG] Sample ZIP paths (first 20):", flush=True)
            for path in file_list[:20]:
                print(f"  {path}", flush=True)
            if len(file_list) > 20:
                print(f"  ... and {len(file_list) - 20} more files", flush=True)
            
            # Check for 100.v1i.folder structure in ZIP
            roboflow_paths = [p for p in file_list if '100.v1i.folder' in p]
            if roboflow_paths:
                print(f"[DEBUG] Found {len(roboflow_paths)} files with '100.v1i.folder' in path", flush=True)
                # Check if they're in train/, valid/, or test/ subdirectories
                train_paths = [p for p in roboflow_paths if '/train/' in p or '\\train\\' in p]
                valid_paths = [p for p in roboflow_paths if '/valid/' in p or '\\valid\\' in p or '/val/' in p or '\\val\\' in p]
                test_paths = [p for p in roboflow_paths if '/test/' in p or '\\test\\' in p]
                print(f"[DEBUG] Roboflow structure in ZIP: train={len(train_paths)}, valid={len(valid_paths)}, test={len(test_paths)}", flush=True)
                if train_paths:
                    print(f"[DEBUG] Sample train path: {train_paths[0]}", flush=True)
                if valid_paths:
                    print(f"[DEBUG] Sample valid path: {valid_paths[0]}", flush=True)
            
            # Check if ZIP has a root folder
            if file_list:
                first_path = file_list[0]
                if '/' in first_path:
                    root_folder = first_path.split('/')[0]
                    print(f"[INFO] ZIP contains root folder: {root_folder}", flush=True)
                    
                    # Check what structure is inside
                    unique_roots = set()
                    for path in file_list:
                        if '/' in path:
                            root = path.split('/')[0]
                            unique_roots.add(root)
                    if len(unique_roots) > 1:
                        print(f"[INFO] ZIP contains multiple root folders: {sorted(unique_roots)}", flush=True)
                    else:
                        print(f"[INFO] ZIP root structure: {root_folder}/", flush=True)
                    
                    # Check for manifest.json first (new standardized approach)
                    manifest_path = None
                    for path in file_list:
                        if 'manifest.json' in path:
                            manifest_path = path
                            break
                    
                    # Always extract to training_data directory first
                    training_data_dir = script_dir / "training_data"
                    training_data_dir.mkdir(parents=True, exist_ok=True)
                    zip_ref.extractall(training_data_dir)
                    print(f"[INFO] Extracted ZIP to {training_data_dir}", flush=True)
                    
                    # Read manifest if available (new standardized approach)
                    if manifest_path:
                        manifest_file = training_data_dir / manifest_path
                        if manifest_file.exists():
                            try:
                                import json
                                with open(manifest_file, 'r') as f:
                                    manifest = json.load(f)
                                print(f"[OK] Found manifest.json: format={manifest.get('format')}, root={manifest.get('root_folder')}", flush=True)
                                root_folder = manifest.get('root_folder', 'dataset_organized')
                                organized_dir = training_data_dir / root_folder
                                print(f"[INFO] Using manifest to locate dataset at: {organized_dir}", flush=True)
                            except Exception as e:
                                print(f"[WARN] Could not read manifest: {e}, using auto-detected root", flush=True)
                                organized_dir = training_data_dir / root_folder
                        else:
                            organized_dir = training_data_dir / root_folder
                    else:
                        # No manifest - use detected root folder
                        organized_dir = training_data_dir / root_folder
                        print(f"[INFO] No manifest found, using auto-detected root: {root_folder}", flush=True)
                else:
                    # Files are at root of ZIP - extract to organized_dir
                    zip_ref.extractall(organized_dir)
                    print(f"[INFO] Extracted root files to {organized_dir}", flush=True)
            else:
                print("[ERROR] ZIP file is empty", flush=True)
                return False
        
        # Clean up temporary ZIP
        os.unlink(tmp_zip_path)
        
        # Verify extraction - data.yaml is optional for classification format
        # Check if dataset structure exists (classification, 100.v1i.folder, or YOLO)
        # Also check if structure is nested inside dataset_organized folder
        has_classification = (organized_dir / "classification" / "train").exists()
        has_roboflow = (organized_dir / "100.v1i.folder" / "train").exists()
        has_yolo = (organized_dir / "train" / "images").exists()
        has_data_yaml = (organized_dir / "data.yaml").exists()
        
        # Also check if files are directly in organized_dir (no nested folder)
        # This can happen if ZIP structure is flat
        training_data_dir = script_dir / "training_data"
        if training_data_dir.exists():
            # Check if dataset_organized was created as subfolder
            for item in training_data_dir.iterdir():
                if item.is_dir() and "dataset" in item.name.lower():
                    nested_organized = item
                    nested_classification = (nested_organized / "classification" / "train").exists()
                    nested_roboflow = (nested_organized / "100.v1i.folder" / "train").exists()
                    nested_yolo = (nested_organized / "train" / "images").exists()
                    
                    if nested_classification or nested_roboflow or nested_yolo:
                        print(f"[INFO] Found dataset in nested folder: {nested_organized}", flush=True)
                        organized_dir = nested_organized
                        has_classification = nested_classification
                        has_roboflow = nested_roboflow
                        has_yolo = nested_yolo
                        has_data_yaml = (nested_organized / "data.yaml").exists()
                        break
        
        if has_classification or has_roboflow or has_yolo:
            print(f"[OK] Dataset extracted successfully to {organized_dir}", flush=True)
            if has_data_yaml:
                print(f"[OK] Found data.yaml (optional for classification)", flush=True)
            else:
                print(f"[INFO] No data.yaml found, but dataset structure exists (will detect classes from folders)", flush=True)
            print(f"[DEBUG] Structure found:", flush=True)
            print(f"  classification/train: {'✓' if has_classification else '✗'}", flush=True)
            print(f"  100.v1i.folder/train: {'✓' if has_roboflow else '✗'}", flush=True)
            print(f"  train/images: {'✓' if has_yolo else '✗'}", flush=True)
            logger.info(f"Dataset extracted to {organized_dir}")
            return True
        else:
            print("[ERROR] Dataset extracted but no valid structure found", flush=True)
            print(f"[DEBUG] Checked in: {organized_dir}", flush=True)
            print(f"[DEBUG] organized_dir exists: {organized_dir.exists()}", flush=True)
            print(f"[DEBUG] Checked for:", flush=True)
            print(f"  classification/train: {'✓' if has_classification else '✗'} ({organized_dir / 'classification' / 'train'})", flush=True)
            print(f"  100.v1i.folder/train: {'✓' if has_roboflow else '✗'} ({organized_dir / '100.v1i.folder' / 'train'})", flush=True)
            print(f"  train/images: {'✓' if has_yolo else '✗'} ({organized_dir / 'train' / 'images'})", flush=True)
            print(f"  data.yaml: {'✓' if has_data_yaml else '✗'} (optional)", flush=True)
            
            # List what was actually extracted - check multiple locations
            locations_to_check = [
                organized_dir,
                script_dir / "training_data" / "dataset_organized",
                script_dir / "training_data",
            ]
            
            for check_dir in locations_to_check:
                if check_dir.exists():
                    print(f"[DEBUG] Contents of {check_dir}:", flush=True)
                    try:
                        items = list(check_dir.iterdir())
                        if items:
                            for item in items[:20]:  # Limit to first 20 items
                                item_type = 'dir' if item.is_dir() else 'file'
                                size_info = ""
                                if item.is_file():
                                    try:
                                        size_info = f" ({item.stat().st_size} bytes)"
                                    except:
                                        pass
                                print(f"  - {item.name} ({item_type}){size_info}", flush=True)
                            if len(items) > 20:
                                print(f"  ... and {len(items) - 20} more items", flush=True)
                        else:
                            print(f"  (empty directory)", flush=True)
                    except Exception as e:
                        print(f"  (error listing: {e})", flush=True)
                    
                    # Also check for nested dataset_organized
                    if check_dir.name != "dataset_organized":
                        nested = check_dir / "dataset_organized"
                        if nested.exists():
                            print(f"[DEBUG] Found nested dataset_organized at: {nested}", flush=True)
                            nested_class = (nested / "classification" / "train").exists()
                            nested_robo = (nested / "100.v1i.folder" / "train").exists()
                            nested_yolo = (nested / "train" / "images").exists()
                            print(f"  classification/train: {'✓' if nested_class else '✗'}", flush=True)
                            print(f"  100.v1i.folder/train: {'✓' if nested_robo else '✗'}", flush=True)
                            print(f"  train/images: {'✓' if nested_yolo else '✗'}", flush=True)
                            
                            # If found here, update organized_dir
                            if nested_class or nested_robo or nested_yolo:
                                organized_dir = nested
                                print(f"[INFO] Using nested dataset_organized at: {organized_dir}", flush=True)
                                logger.info(f"Found dataset in nested location: {organized_dir}")
                                return True
            
            logger.error("Dataset extracted but no valid structure found")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error downloading dataset: {e}", flush=True)
        logger.error(f"Network error downloading dataset: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error downloading dataset: {e}", flush=True)
        logger.error(f"Error downloading dataset: {e}")
        return False

def create_combined_dataset(logger, job_id=None):
    """Create combined dataset from original and collected data"""
    import sys
    print("Creating combined dataset...", flush=True)
    logger.info("Creating combined dataset...")
    
    # Directories - use parent directory (Proto1/)
    script_dir = Path(__file__).resolve().parent
    
    # PRIORITY 0: Download dataset from server if not available locally (for Heroku)
    organized_dir = script_dir / "training_data" / "dataset_organized"
    
    # CRITICAL FIX: ALWAYS remove existing dataset directory FIRST to ensure fresh dataset
    # This prevents using stale/cached datasets from previous training runs
    # The old dataset might have wrong image counts (e.g., 326 instead of 1106)
    if organized_dir.exists():
        try:
            print(f"[INFO] Removing existing dataset directory to ensure fresh download...", flush=True)
            print(f"[INFO] This prevents using stale datasets from previous training runs", flush=True)
            shutil.rmtree(organized_dir)
            print(f"[OK] Existing dataset directory removed", flush=True)
            logger.info("Removed existing dataset directory to ensure fresh download")
        except Exception as e:
            print(f"[WARN] Could not remove existing dataset directory: {e}", flush=True)
            logger.warning(f"Could not remove existing dataset directory: {e}")
            # Try to at least remove classification directory
            classification_old = organized_dir / "classification"
            if classification_old.exists():
                try:
                    shutil.rmtree(classification_old)
                    print(f"[OK] Removed old classification directory", flush=True)
                except:
                    pass
    
    # Now check if we need to download (will always be True after removal)
    organized_yaml_check = organized_dir / "data.yaml"
    organized_train_images_check = organized_dir / "train" / "images"  # YOLO format
    classification_train_check = organized_dir / "classification" / "train"  # Classification format
    
    # After removal, dataset won't exist, so we always need to download
    needs_download = True
    
    if needs_download:
        print("[INFO] Downloading fresh dataset (DATASET_ZIP_URL or PHP)...", flush=True)
        print(f"[INFO] Prefer Google Drive if DATASET_ZIP_URL is set on Railway", flush=True)
        logger.info("Downloading fresh dataset from server")
        
        # Get job_id from args if available
        job_id_for_download = None
        try:
            import sys
            for arg in sys.argv:
                if arg.startswith('--job_id'):
                    job_id_for_download = int(arg.split('=')[1] if '=' in arg else sys.argv[sys.argv.index(arg) + 1])
                    break
        except:
            pass
        
        download_success = download_dataset_from_server(script_dir, logger, job_id=job_id_for_download)
        if not download_success:
            print("[ERROR] Could not download dataset from server!", flush=True)
            print("[ERROR] Please ensure a dataset is uploaded on the server.", flush=True)
            logger.error("Could not download dataset from server")
            # Don't continue if download failed - we need the dataset
            raise FileNotFoundError("Dataset download failed. Please upload a dataset first.")
        else:
            print("[OK] Dataset downloaded successfully", flush=True)
            logger.info("Dataset downloaded successfully")
            
            # Refresh paths after download
            organized_dir = script_dir / "training_data" / "dataset_organized"
            organized_train_images_check = organized_dir / "train" / "images"  # YOLO format
            classification_train_check = organized_dir / "classification" / "train"  # Classification format
            
            # Verify images exist after download (check both formats)
            image_count_after = 0
            has_yolo_after = False
            has_classification_after = False
            
            # Check YOLO format
            if organized_train_images_check.exists():
                yolo_count = len(list(organized_train_images_check.glob('*.jpg')) + 
                               list(organized_train_images_check.glob('*.jpeg')) + 
                               list(organized_train_images_check.glob('*.png')))
                if yolo_count > 0:
                    image_count_after += yolo_count
                    has_yolo_after = True
                    print(f"[INFO] Found YOLO format: {yolo_count} images in train/images", flush=True)
            
            # Check classification format (from database export)
            if classification_train_check.exists():
                classification_count = 0
                for class_dir in classification_train_check.iterdir():
                    if class_dir.is_dir():
                        classification_count += len(list(class_dir.glob('*.jpg')) + 
                                                  list(class_dir.glob('*.jpeg')) + 
                                                  list(class_dir.glob('*.png')))
                if classification_count > 0:
                    image_count_after += classification_count
                    has_classification_after = True
                    print(f"[INFO] Found classification format: {classification_count} images in classification/train/", flush=True)
            
            # Check 100.v1i.folder structure (original Roboflow dataset)
            roboflow_dir = organized_dir / "100.v1i.folder"
            has_roboflow = False
            if roboflow_dir.exists():
                roboflow_train = roboflow_dir / "train"
                roboflow_valid = roboflow_dir / "valid"
                roboflow_count = 0
                if roboflow_train.exists():
                    for class_dir in roboflow_train.iterdir():
                        if class_dir.is_dir():
                            roboflow_count += len(list(class_dir.glob('*.jpg')) + 
                                                list(class_dir.glob('*.jpeg')) + 
                                                list(class_dir.glob('*.png')))
                if roboflow_valid.exists():
                    for class_dir in roboflow_valid.iterdir():
                        if class_dir.is_dir():
                            roboflow_count += len(list(class_dir.glob('*.jpg')) + 
                                                list(class_dir.glob('*.jpeg')) + 
                                                list(class_dir.glob('*.png')))
                if roboflow_count > 0:
                    image_count_after += roboflow_count
                    has_roboflow = True
                    print(f"[INFO] Found 100.v1i.folder format: {roboflow_count} images in 100.v1i.folder/", flush=True)
            
            print(f"[INFO] Dataset verification: {image_count_after} total images found after download", flush=True)
            print(f"[INFO] Formats detected: YOLO={has_yolo_after}, Classification={has_classification_after}, Roboflow={has_roboflow}", flush=True)
            
            if image_count_after == 0:
                print("[ERROR] Dataset downloaded but no images found! Check dataset structure on server.", flush=True)
                print("[ERROR] The dataset ZIP file may be empty or have incorrect structure.", flush=True)
                logger.error("Dataset downloaded but contains no images")
                raise FileNotFoundError("Dataset downloaded but contains no images. Please check the dataset on the server.")
            
            if not has_yolo_after and not has_classification_after and not has_roboflow:
                print("[ERROR] Dataset downloaded but no valid format found!", flush=True)
                print("[ERROR] Expected: train/images/ (YOLO), classification/train/ (classification), or 100.v1i.folder/ (Roboflow)", flush=True)
                logger.error("Dataset downloaded but no valid format found")
                raise FileNotFoundError("Dataset downloaded but no valid format found. Expected train/images/ (YOLO), classification/train/ (classification), or 100.v1i.folder/ (Roboflow).")
    
    # PRIORITY 1: Check for organized dataset (data.yaml is optional for classification)
    organized_dir = script_dir / "training_data" / "dataset_organized"
    organized_yaml = organized_dir / "data.yaml"
    
    # Try to load classes from data.yaml if it exists (optional)
    pest_classes = None
    if organized_yaml.exists():
        pest_classes = load_classes_from_yaml(organized_yaml)
        if pest_classes:
            print(f"[OK] Found data.yaml with {len(pest_classes)} classes: {pest_classes}", flush=True)
            logger.info(f"Loaded {len(pest_classes)} classes from data.yaml: {pest_classes}")
        else:
            print(f"[INFO] data.yaml exists but could not parse classes, will detect from folders", flush=True)
            logger.info("data.yaml exists but could not parse classes, will detect from folders")
    else:
        print(f"[INFO] No data.yaml found (optional for classification), will detect classes from folder structure", flush=True)
        logger.info("No data.yaml found, will detect classes from folder structure")
    
    # PRIORITY 0.3: Prefer already-normalized classification/ (from dataset_source)
    # before re-converting 100.v1i.folder — avoids duplicate hopper aliases.
    classification_train_dir = organized_dir / "classification" / "train"
    classification_val_dir = organized_dir / "classification" / "val"

    def _count_images_in_class_dirs(root: Path) -> int:
        if not root.exists():
            return 0
        total = 0
        for class_dir in root.iterdir():
            if class_dir.is_dir():
                total += len(
                    list(class_dir.glob("*.jpg"))
                    + list(class_dir.glob("*.jpeg"))
                    + list(class_dir.glob("*.png"))
                )
        return total

    train_images_count = _count_images_in_class_dirs(classification_train_dir)
    val_images_count = _count_images_in_class_dirs(classification_val_dir)

    if train_images_count > 0 or val_images_count > 0:
        print(f"[OK] Found classification format dataset", flush=True)
        print(f"  Train images: {train_images_count}, Val images: {val_images_count}", flush=True)
        logger.info(
            f"Using existing classification format dataset: {train_images_count} train, {val_images_count} val"
        )
        if pest_classes:
            print(f"  Using classes from data.yaml: {pest_classes}", flush=True)
            return classification_train_dir, classification_val_dir, pest_classes
        detected_classes = sorted(
            [d.name for d in classification_train_dir.iterdir() if d.is_dir()]
        )
        if detected_classes:
            print(f"  Detected classes from directory: {detected_classes}", flush=True)
            logger.info(f"Detected classes from directory: {detected_classes}")
            return classification_train_dir, classification_val_dir, detected_classes

    # PRIORITY 0.4: Convert 100.v1i.folder only when classification/ is missing
    roboflow_dir = organized_dir / "100.v1i.folder"
    if roboflow_dir.exists():
        roboflow_train = roboflow_dir / "train"
        roboflow_valid = roboflow_dir / "valid"

        if roboflow_train.exists() or roboflow_valid.exists():
            from dataset_source import canonical_class_name

            print(
                "[INFO] Found 100.v1i.folder structure, converting to classification format...",
                flush=True,
            )
            logger.info("Found 100.v1i.folder structure, converting to classification format")

            # Rebuild cleanly so alias folders cannot coexist
            classification_root = organized_dir / "classification"
            if classification_root.exists():
                shutil.rmtree(classification_root)
            classification_train_dir = classification_root / "train"
            classification_val_dir = classification_root / "val"
            classification_train_dir.mkdir(parents=True, exist_ok=True)
            classification_val_dir.mkdir(parents=True, exist_ok=True)

            train_count = 0
            val_count = 0
            classes_found = []

            def _convert_split(src_root: Path, dst_root: Path) -> int:
                copied = 0
                if not src_root.exists():
                    return 0
                for roboflow_class_dir in src_root.iterdir():
                    if not roboflow_class_dir.is_dir():
                        continue
                    system_class_name = canonical_class_name(roboflow_class_dir.name)
                    if system_class_name not in classes_found:
                        classes_found.append(system_class_name)
                    system_dir = dst_root / system_class_name
                    system_dir.mkdir(exist_ok=True)
                    image_files = (
                        list(roboflow_class_dir.glob("*.jpg"))
                        + list(roboflow_class_dir.glob("*.jpeg"))
                        + list(roboflow_class_dir.glob("*.png"))
                    )
                    for img_file in image_files:
                        dest = system_dir / img_file.name
                        if not dest.exists():
                            shutil.copy2(img_file, dest)
                            copied += 1
                return copied

            train_count = _convert_split(roboflow_train, classification_train_dir)
            val_count = _convert_split(roboflow_valid, classification_val_dir)

            print(f"[DEBUG] Classes found during conversion: {classes_found}", flush=True)

            if train_count > 0 or val_count > 0:
                detected_classes = sorted(
                    [d.name for d in classification_train_dir.iterdir() if d.is_dir()]
                )
                # Keep data.yaml in sync with canonical classes
                yaml_path = organized_dir / "data.yaml"
                with open(yaml_path, "w", encoding="utf-8") as f:
                    f.write("names:\n")
                    for i, c in enumerate(detected_classes):
                        f.write(f"  {i}: {c}\n")
                    f.write(f"nc: {len(detected_classes)}\n")

                print(
                    f"[OK] Converted 100.v1i.folder to classification format: {train_count} train, {val_count} val",
                    flush=True,
                )
                logger.info(
                    f"Converted 100.v1i.folder to classification format: {train_count} train, {val_count} val"
                )
                if detected_classes:
                    print(f"  Detected classes: {detected_classes}", flush=True)
                    logger.info(f"Detected classes: {detected_classes}")
                    return classification_train_dir, classification_val_dir, detected_classes
    
    # Check if organized dataset exists (from smart import)
    organized_train_images = organized_dir / "train" / "images"
    organized_train_labels = organized_dir / "train" / "labels"
    organized_val_images = organized_dir / "valid" / "images"
    organized_val_labels = organized_dir / "valid" / "labels"
    
    # PRIORITY 1A: Use database to get images (most reliable - images are stored with pest_class)
    if organized_yaml.exists() and pest_classes:
        print(f"[INFO] Checking database for imported images...", flush=True)
        logger.info("Checking database for imported images")
        
        # Only try database if pymysql is available (local development)
        if not PYMYSQL_AVAILABLE or not DB_CONFIG:
            print("[INFO] Skipping database image lookup (using PHP API gateway)", flush=True)
            db_images = []
        else:
            try:
                conn = pymysql.connect(**DB_CONFIG)
                cursor = conn.cursor()
                
                # Get all images from training_images table
                cursor.execute("SELECT file_path, pest_class FROM training_images WHERE is_verified = 1")
                db_images = cursor.fetchall()
                conn.close()
            except Exception as e:
                print(f"[WARNING] Could not access database for images: {e}", flush=True)
                db_images = []
            
            if db_images and len(db_images) > 0:
                print(f"[OK] Found {len(db_images)} images in database", flush=True)
                logger.info(f"Found {len(db_images)} images in database")
                
                # Create classification-ready dataset structure
                classification_train_dir = organized_dir / "classification" / "train"
                classification_val_dir = organized_dir / "classification" / "val"
                
                # Create class folders
                for split_dir in [classification_train_dir, classification_val_dir]:
                    split_dir.mkdir(parents=True, exist_ok=True)
                    for class_name in pest_classes:
                        (split_dir / class_name).mkdir(exist_ok=True)
                
                # Normalize pest_class names to match YAML classes (handle variations)
                def normalize_class_name(db_class):
                    """Normalize database class name to match YAML class names"""
                    db_class_lower = db_class.lower().replace(' ', '_').replace('-', '_')
                    # Try exact match first
                    for yaml_class in pest_classes:
                        if db_class_lower == yaml_class.lower():
                            return yaml_class
                    # Try partial match
                    for yaml_class in pest_classes:
                        if yaml_class.lower() in db_class_lower or db_class_lower in yaml_class.lower():
                            return yaml_class
                    return None
                
                # Reorganize images from database
                import random
                random.seed(42)
                train_count = 0
                val_count = 0
                
                for file_path, db_pest_class in db_images:
                    img_path = Path(file_path)
                    if not img_path.exists():
                        # Try relative path from script directory
                        img_path = script_dir / file_path.lstrip('/')
                        if not img_path.exists():
                            continue
                    
                    # Normalize class name
                    class_name = normalize_class_name(db_pest_class)
                    if not class_name:
                        logger.warning(f"Could not map database class '{db_pest_class}' to YAML classes")
                        continue
                    
                    # Split 80% train, 20% val
                    is_train = random.random() < 0.8
                    dest_dir = classification_train_dir if is_train else classification_val_dir
                    dest = dest_dir / class_name / img_path.name
                    
                    if not dest.exists():
                        try:
                            shutil.copy2(img_path, dest)
                            if is_train:
                                train_count += 1
                            else:
                                val_count += 1
                        except Exception as e:
                            logger.warning(f"Error copying {img_path}: {e}")
                            continue
                
                if train_count > 0 or val_count > 0:
                    print(f"[OK] Reorganized {len(db_images)} images from database: {train_count} train, {val_count} val", flush=True)
                    logger.info(f"Reorganized {len(db_images)} images from database: {train_count} train, {val_count} val")
                    print(f"  Using reorganized dataset at: {classification_train_dir}", flush=True)
                    sys.stdout.flush()
                    return classification_train_dir, classification_val_dir, pest_classes
                else:
                    print(f"[WARN] Could not copy images from database, trying YOLO labels...", flush=True)
                    logger.warning("Could not copy images from database, trying YOLO labels")
            else:
                print(f"[INFO] No images found in database, trying YOLO format...", flush=True)
                logger.info("No images found in database, trying YOLO format")
    
    # PRIORITY 1B: Reorganize from YOLO format (if database method didn't work)
    # Check if we have the necessary directories for YOLO reorganization
    has_yolo_structure = (
        organized_yaml.exists() and 
        pest_classes and 
        (organized_train_images.exists() or organized_val_images.exists())
    )
    
    if has_yolo_structure:
        print(f"[INFO] Found organized dataset from import, reorganizing from YOLO format...", flush=True)
        logger.info("Reorganizing organized dataset from YOLO format into class folders")
        
        # Debug: Check what directories actually exist
        print(f"[DEBUG] Checking YOLO structure:", flush=True)
        print(f"  data.yaml: {organized_yaml.exists()}", flush=True)
        print(f"  train/images: {organized_train_images.exists()}", flush=True)
        print(f"  train/labels: {organized_train_labels.exists()}", flush=True)
        print(f"  valid/images: {organized_val_images.exists()}", flush=True)
        print(f"  valid/labels: {organized_val_labels.exists()}", flush=True)
        
        if organized_train_images.exists():
            train_img_count = len(list(organized_train_images.glob('*.jpg')) + 
                                 list(organized_train_images.glob('*.jpeg')) + 
                                 list(organized_train_images.glob('*.png')))
            print(f"  train/images count: {train_img_count}", flush=True)
        
        if organized_val_images.exists():
            val_img_count = len(list(organized_val_images.glob('*.jpg')) + 
                              list(organized_val_images.glob('*.jpeg')) + 
                              list(organized_val_images.glob('*.png')))
            print(f"  valid/images count: {val_img_count}", flush=True)
        
        # Create classification-ready dataset structure
        classification_train_dir = organized_dir / "classification" / "train"
        classification_val_dir = organized_dir / "classification" / "val"
        
        # Create class folders
        for split_dir in [classification_train_dir, classification_val_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
            for class_name in pest_classes:
                (split_dir / class_name).mkdir(exist_ok=True)
        
        # Function to reorganize images based on YOLO labels
        def reorganize_from_yolo(images_dir, labels_dir, output_dir, split_name):
            if not images_dir.exists():
                print(f"  [WARN] Images directory not found: {images_dir}", flush=True)
                return 0
            if not labels_dir.exists():
                print(f"  [WARN] Labels directory not found: {labels_dir}", flush=True)
                return 0
            
            reorganized_count = 0
            skipped_no_label = 0
            skipped_invalid_label = 0
            skipped_duplicate = 0
            skipped_invalid_class = 0
            
            images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
            print(f"  [INFO] Found {len(images)} images in {images_dir}", flush=True)
            
            for img_path in images:
                label_path = labels_dir / (img_path.stem + '.txt')
                if not label_path.exists():
                    skipped_no_label += 1
                    continue
                
                # Read first line of label to get class index
                try:
                    with open(label_path, 'r') as f:
                        first_line = f.readline().strip()
                    if not first_line:
                        skipped_invalid_label += 1
                        continue
                    
                    parts = first_line.split()
                    if len(parts) < 5:
                        skipped_invalid_label += 1
                        continue
                    
                    class_index = int(parts[0])
                    # Map class index to class name (assuming indices match YAML order)
                    if class_index >= len(pest_classes):
                        skipped_invalid_class += 1
                        continue
                    
                    class_name = pest_classes[class_index]
                    # Copy image to class folder
                    dest = output_dir / class_name / img_path.name
                    if dest.exists():  # Avoid duplicates
                        skipped_duplicate += 1
                        continue
                    
                    shutil.copy2(img_path, dest)
                    reorganized_count += 1
                except Exception as e:
                    skipped_invalid_label += 1
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
            
            # Print detailed statistics
            total_processed = len(images)
            total_skipped = skipped_no_label + skipped_invalid_label + skipped_duplicate + skipped_invalid_class
            print(f"  [INFO] {split_name}: {reorganized_count}/{total_processed} images reorganized", flush=True)
            if total_skipped > 0:
                print(f"  [INFO]   Skipped: {skipped_no_label} (no label), {skipped_invalid_label} (invalid label), {skipped_duplicate} (duplicate), {skipped_invalid_class} (invalid class)", flush=True)
            
            return reorganized_count
        
        # Reorganize train and val sets
        train_count = reorganize_from_yolo(organized_train_images, organized_train_labels, classification_train_dir, "train")
        val_count = reorganize_from_yolo(organized_val_images, organized_val_labels, classification_val_dir, "val")
        
        # Also include test images in training set (optional - use all available data)
        organized_test_images = organized_dir / "test" / "images"
        organized_test_labels = organized_dir / "test" / "labels"
        test_count = 0
        if organized_test_images.exists() and organized_test_labels.exists():
            # Add test images to training set (not validation, to maximize training data)
            test_count = reorganize_from_yolo(organized_test_images, organized_test_labels, classification_train_dir, "test")
            if test_count > 0:
                print(f"[INFO] Added {test_count} test images to training set (using all available data)", flush=True)
                logger.info(f"Added {test_count} test images to training set")
        
        if train_count > 0 or val_count > 0:
            total_train = train_count + test_count
            print(f"[OK] Reorganized dataset from YOLO: {total_train} train images ({train_count} train + {test_count} test), {val_count} val images", flush=True)
            logger.info(f"Reorganized dataset from YOLO: {total_train} train images, {val_count} val images")
            print(f"  Total images used: {total_train + val_count} (out of {train_count + val_count + test_count} available)", flush=True)
            print(f"  Using reorganized dataset at: {classification_train_dir}", flush=True)
            sys.stdout.flush()
            return classification_train_dir, classification_val_dir, pest_classes
        else:
            print(f"[WARN] No images found in organized dataset after reorganization", flush=True)
            print(f"[DEBUG] Train count: {train_count}, Val count: {val_count}, Test count: {test_count}", flush=True)
            if organized_train_images.exists():
                all_train_imgs = list(organized_train_images.glob('*.jpg')) + list(organized_train_images.glob('*.jpeg')) + list(organized_train_images.glob('*.png'))
                print(f"[DEBUG] Train images dir contains {len(all_train_imgs)} image files", flush=True)
                if len(all_train_imgs) == 0:
                    all_items = list(organized_train_images.glob('*'))
                    print(f"[DEBUG] Train images dir contains {len(all_items)} total items (may include subdirectories)", flush=True)
            if organized_train_labels.exists():
                all_train_lbls = list(organized_train_labels.glob('*.txt'))
                print(f"[DEBUG] Train labels dir contains {len(all_train_lbls)} label files", flush=True)
            logger.warning("No images found in organized dataset, falling back")
            
            # If dataset exists but has no images, try to download again
            if organized_dir.exists() and organized_yaml.exists():
                print(f"[INFO] Dataset directory exists but has no images. Attempting to re-download...", flush=True)
                logger.info("Dataset directory exists but empty, attempting re-download")
                
                # Remove empty dataset directory first
                try:
                    if organized_dir.exists():
                        print(f"[INFO] Removing empty dataset directory: {organized_dir}", flush=True)
                        shutil.rmtree(organized_dir)
                        print(f"[OK] Empty dataset directory removed", flush=True)
                except Exception as e:
                    print(f"[WARN] Could not remove empty dataset directory: {e}", flush=True)
                    logger.warning(f"Could not remove empty dataset directory: {e}")
                
                # Get job_id from args if available
                job_id_for_download = None
                try:
                    import sys
                    for arg in sys.argv:
                        if arg.startswith('--job_id'):
                            job_id_for_download = int(arg.split('=')[1] if '=' in arg else sys.argv[sys.argv.index(arg) + 1])
                            break
                except:
                    pass
                
                download_success = download_dataset_from_server(script_dir, logger, job_id=job_id_for_download)
                if download_success:
                    # Refresh paths after download
                    organized_dir = script_dir / "training_data" / "dataset_organized"
                    organized_train_images = organized_dir / "train" / "images"
                    organized_train_labels = organized_dir / "train" / "labels"
                    organized_val_images = organized_dir / "valid" / "images"
                    organized_val_labels = organized_dir / "valid" / "labels"
                    classification_train_dir = organized_dir / "classification" / "train"
                    classification_val_dir = organized_dir / "classification" / "val"
                    classification_train_dir.mkdir(parents=True, exist_ok=True)
                    classification_val_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Try reorganization again after download
                    train_count = reorganize_from_yolo(organized_train_images, organized_train_labels, classification_train_dir, "train")
                    val_count = reorganize_from_yolo(organized_val_images, organized_val_labels, classification_val_dir, "val")
                    if train_count > 0 or val_count > 0:
                        total_train = train_count + test_count
                        print(f"[OK] Reorganized dataset after re-download: {total_train} train, {val_count} val images", flush=True)
                        logger.info(f"Reorganized dataset after re-download: {total_train} train, {val_count} val images")
                        return classification_train_dir, classification_val_dir, pest_classes
                    else:
                        print(f"[ERROR] Re-download succeeded but still no images found after reorganization", flush=True)
                        logger.error("Re-download succeeded but still no images found after reorganization")
                else:
                    print(f"[ERROR] Re-download failed. Cannot proceed without dataset.", flush=True)
                    logger.error("Re-download failed. Cannot proceed without dataset.")
                    raise FileNotFoundError("Dataset download failed. Please upload a dataset to the server first.")
    
    # PRIORITY 2: Fallback to old dataset structure (only if organized dataset doesn't exist or has no images)
    original_train_dir = script_dir / "ml_training" / "datasets" / "processed" / "train"
    original_val_dir = script_dir / "ml_training" / "datasets" / "processed" / "val"
    collected_data_dir = script_dir / "ml_training" / "datasets" / "auto_collected"
    combined_dir = script_dir / "ml_training" / "datasets" / "combined"
    
    # Fallback to hardcoded classes if yaml not found
    if not pest_classes:
        print("Warning: data.yaml not found, using default classes", flush=True)
        logger.warning("data.yaml not found, using default classes")
        pest_classes = ['leptocorisa_oratorius', 'nephotettix_virescens', 'nilaparvata_lugens', 'scotinophara_coarctata', 'scirpophaga_incertulas']
    else:
        print(f"Loaded {len(pest_classes)} classes from data.yaml: {pest_classes}", flush=True)
        logger.info(f"Loaded {len(pest_classes)} classes from data.yaml: {pest_classes}")
    
    print(f"Checking directories...", flush=True)
    print(f"  Train dir: {original_train_dir} (exists: {original_train_dir.exists()})", flush=True)
    print(f"  Val dir: {original_val_dir} (exists: {original_val_dir.exists()})", flush=True)
    sys.stdout.flush()
    
    # If processed directories don't exist, use them directly
    if not original_train_dir.exists():
        print(f"WARNING: {original_train_dir} not found, checking alternatives...", flush=True)
        # Try alternative paths
        alt_paths = [
            script_dir / "datasets" / "processed" / "train",
            script_dir / "training_data" / "train",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                original_train_dir = alt_path
                original_val_dir = script_dir / alt_path.parent / "val"
                print(f"Using alternative: {original_train_dir}", flush=True)
                break
        else:
            raise FileNotFoundError(f"Training data directory not found. Checked: {original_train_dir}")
    
    # SIMPLIFIED: Use existing directories directly (no copying = much faster!)
    if original_train_dir.exists():
        if original_val_dir.exists():
            print(f"[OK] Using existing directories directly", flush=True)
            print(f"  Train: {original_train_dir}", flush=True)
            print(f"  Val: {original_val_dir}", flush=True)
            sys.stdout.flush()
            return original_train_dir, original_val_dir, pest_classes
        else:
            print(f"[WARN] Val dir missing, using train for both", flush=True)
            return original_train_dir, original_train_dir, pest_classes
    
    # Only create combined dataset if original doesn't exist
    print("Creating combined dataset structure...", flush=True)
    combined_train_dir = combined_dir / "train"
    combined_val_dir = combined_dir / "val"
    
    for split_dir in [combined_train_dir, combined_val_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for class_name in pest_classes:
            (split_dir / class_name).mkdir(exist_ok=True)
    
    # Copy original training data
    logger.info("Copying original training data...")
    for class_name in pest_classes:
        # Copy original train data
        original_train_class_dir = original_train_dir / class_name
        if original_train_class_dir.exists():
            for img_file in original_train_class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest = combined_train_dir / class_name / f"original_{img_file.name}"
                    shutil.copy2(img_file, dest)
        
        # Copy original val data
        original_val_class_dir = original_val_dir / class_name
        if original_val_class_dir.exists():
            for img_file in original_val_class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest = combined_val_dir / class_name / f"original_{img_file.name}"
                    shutil.copy2(img_file, dest)
    
    # Copy collected data (80% train, 20% val)
    logger.info("Adding auto-collected data...")
    import random
    random.seed(42)
    
    for class_name in pest_classes:
        collected_class_dir = collected_data_dir / class_name
        if collected_class_dir.exists():
            images = [f for f in collected_class_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            if len(images) > 0:
                # Shuffle and split collected data
                random.shuffle(images)
                split_point = int(len(images) * 0.8)
                
                train_images = images[:split_point]
                val_images = images[split_point:]
                
                # Copy to train
                for img in train_images:
                    dest = combined_train_dir / class_name / f"collected_{img.name}"
                    shutil.copy2(img, dest)
                
                # Copy to val
                for img in val_images:
                    dest = combined_val_dir / class_name / f"collected_{img.name}"
                    shutil.copy2(img, dest)
                
                logger.info(f"{class_name}: +{len(images)} collected images ({len(train_images)} train, {len(val_images)} val)")
    
    return combined_train_dir, combined_val_dir, pest_classes

# YOLO training function removed - reverting to classification

def main():
    """Main training function - Classification using ResNet18"""
    parser = argparse.ArgumentParser(description='Admin Training Script - YOLO Object Detection')
    parser.add_argument('--job_id', type=int, required=True, help='Training job ID')
    # Get default epochs from environment variable
    default_epochs = int(os.getenv('DEFAULT_EPOCHS', '3'))
    parser.add_argument('--epochs', type=int, default=default_epochs, help=f'Number of epochs (default: {default_epochs})')
    parser.add_argument('--batch_size', type=int, default=int(os.getenv('DEFAULT_BATCH_SIZE', '16')), help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (not used for YOLO)')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = AdminTrainingLogger(args.job_id, DB_CONFIG)
    
    try:
        # Force output immediately
        import sys
        print("=" * 50, flush=True)
        print("PEST DETECTION TRAINING", flush=True)
        print("=" * 50, flush=True)
        print(f"Job ID: {args.job_id}", flush=True)
        print(f"Epochs: {args.epochs}", flush=True)
        print(f"Batch Size: {args.batch_size}", flush=True)
        print(f"Learning Rate: {args.learning_rate}", flush=True)
        print("=" * 50, flush=True)
        sys.stdout.flush()
        
        logger.info(f"Starting training job {args.job_id}")
        logger.info(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
        
        # Update job status to running
        update_job_status(args.job_id, 'running')
        
        # Create combined dataset - with error handling
        try:
            print("Creating dataset...", flush=True)
            sys.stdout.flush()
            # Pass job_id to create_combined_dataset so it can pass to download_dataset_from_server
            train_dir, val_dir, classes_from_yaml = create_combined_dataset(logger, job_id=args.job_id)
            print(f"[OK] Dataset created: Train={train_dir}, Val={val_dir}", flush=True)
            if classes_from_yaml:
                print(f"[OK] Classes from data.yaml: {len(classes_from_yaml)} classes - {classes_from_yaml}", flush=True)
            sys.stdout.flush()
        except Exception as e:
            import traceback
            error_msg = f"Failed to create dataset: {str(e)}"
            traceback_str = traceback.format_exc()
            print(f"ERROR: {error_msg}", flush=True)
            print(f"Traceback: {traceback_str}", flush=True)
            logger.error(error_msg)
            logger.error(traceback_str)
            update_job_status(args.job_id, 'failed', error_msg)
            sys.exit(1)
        
        # Get data transforms
        print("Creating model trainer...", flush=True)
        sys.stdout.flush()
        trainer = ModelTrainer(args.job_id, {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        }, logger)
        
        print("Getting data transforms...", flush=True)
        sys.stdout.flush()
        train_transforms, val_transforms = trainer.get_data_transforms()
        
        # Create datasets
        print("Loading datasets...", flush=True)
        print(f"  Train directory: {train_dir}", flush=True)
        print(f"  Val directory: {val_dir}", flush=True)
        if classes_from_yaml:
            print(f"  Classes from data.yaml: {len(classes_from_yaml)} classes - {classes_from_yaml}", flush=True)
        sys.stdout.flush()
        logger.info("Loading datasets...")
        logger.info(f"Train directory: {train_dir}")
        logger.info(f"Val directory: {val_dir}")
        if classes_from_yaml:
            logger.info(f"Classes from data.yaml: {classes_from_yaml}")
        
        # Pass classes from YAML to dataset class so it uses the correct number of classes
        train_dataset = EnhancedPestDataset(train_dir, transform=train_transforms, logger=logger, classes_from_yaml=classes_from_yaml)
        val_dataset = EnhancedPestDataset(val_dir, transform=val_transforms, logger=logger, classes_from_yaml=classes_from_yaml)
        
        print(f"[OK] Datasets loaded: Train={len(train_dataset)} samples, Val={len(val_dataset)} samples", flush=True)
        print(f"[INFO] Number of classes detected: {len(train_dataset.classes)}", flush=True)
        print(f"[INFO] Classes: {train_dataset.classes}", flush=True)
        sys.stdout.flush()
        
        # Log dataset statistics
        train_stats = train_dataset.get_statistics()
        val_stats = val_dataset.get_statistics()
        
        logger.info(f"Training dataset: {train_stats}")
        logger.info(f"Validation dataset: {val_stats}")
        logger.info(f"Number of classes: {len(train_dataset.classes)}")
        logger.info(f"Classes: {train_dataset.classes}")
        
        # Start training
        model = trainer.train(train_dataset, val_dataset)
        
        # After training completes, upload the final best model once
        if trainer.best_accuracy > 0:
            logger.info(f"Training completed! Final best accuracy: {trainer.best_accuracy:.2f}%")
            print(f"[OK] Training completed! Best accuracy: {trainer.best_accuracy:.2f}%", flush=True)
            
            script_dir = Path(__file__).resolve().parent
            model_dir = script_dir / "models" / f"job_{args.job_id}"
            onnx_path = model_dir / "best_model.onnx"
            pth_path = model_dir / "best_model.pth"
            
            # Upload the final best model (only once at the end)
            if onnx_path.exists():
                logger.info("Uploading final best model to server...")
                print(f"[INFO] Uploading final best model (accuracy: {trainer.best_accuracy:.2f}%)...", flush=True)
                upload_success = trainer.upload_model_to_server(onnx_path, trainer.best_accuracy, 'onnx')
                
                if upload_success:
                    logger.info("Final model uploaded successfully!")
                    print(f"[OK] Final model uploaded successfully!", flush=True)
                else:
                    # Log model location for manual upload
                    logger.warning(f"Model upload failed, but model is saved locally at: {onnx_path}")
                    logger.warning(f"Model size: {onnx_path.stat().st_size / (1024 * 1024):.2f} MB")
                    logger.warning(f"To manually upload: Copy {onnx_path} to server and register in model_versions table")
                    print(f"[WARN] Final model upload failed, but model is saved locally", flush=True)
                    print(f"[INFO] Model location: {onnx_path}", flush=True)
                    print(f"[INFO] Model size: {onnx_path.stat().st_size / (1024 * 1024):.2f} MB", flush=True)
                    print(f"[INFO] You can manually upload this file to activate the model", flush=True)
                    logger.warning("Final model upload failed, but model is saved locally")
                    print(f"[WARN] Final model upload failed, but model is saved locally", flush=True)
                
                # Copy to standard location for detection API
                standard_model_path = script_dir / "models" / "best.onnx"
                try:
                    shutil.copy2(onnx_path, standard_model_path)
                    logger.info(f"Copied model to standard location: {standard_model_path}")
                    print(f"[OK] Model copied to standard location: best.onnx", flush=True)
                except Exception as e:
                    logger.warning(f"Could not copy model to standard location: {e}")
                    print(f"[WARN] Could not copy model to standard location: {e}", flush=True)
            elif pth_path.exists():
                # Only PyTorch model exists, convert to ONNX first, then upload
                logger.info("Converting final model to ONNX and uploading...")
                print(f"[INFO] Converting final model to ONNX...", flush=True)
                onnx_path = trainer.convert_to_onnx(model, pth_path)
                if onnx_path and onnx_path.exists():
                    logger.info("Uploading final best model to server...")
                    print(f"[INFO] Uploading final best model (accuracy: {trainer.best_accuracy:.2f}%)...", flush=True)
                    upload_success = trainer.upload_model_to_server(onnx_path, trainer.best_accuracy, 'onnx')
                    
                    if upload_success:
                        logger.info("Final model uploaded successfully!")
                        print(f"[OK] Final model uploaded successfully!", flush=True)
                    else:
                        logger.warning("Final model upload failed, but model is saved locally")
                        print(f"[WARN] Final model upload failed, but model is saved locally", flush=True)
                    
                    # Also copy to standard location
                    standard_model_path = script_dir / "models" / "best.onnx"
                    try:
                        shutil.copy2(onnx_path, standard_model_path)
                        logger.info(f"Copied model to standard location: {standard_model_path}")
                        print(f"[OK] Model copied to standard location: best.onnx", flush=True)
                    except Exception as e:
                        logger.warning(f"Could not copy model to standard location: {e}")
                        print(f"[WARN] Could not copy model to standard location: {e}", flush=True)
                else:
                    logger.error("Failed to convert model to ONNX")
                    print(f"[ERROR] Failed to convert model to ONNX", flush=True)
            else:
                logger.error("No model file found after training!")
                print(f"[ERROR] No model file found after training!", flush=True)
        
        # Update job status to completed
        update_job_status(args.job_id, 'completed')
        logger.info(f"Training job {args.job_id} completed successfully!")
        print(f"[OK] Training job {args.job_id} completed!", flush=True)
        
    except Exception as e:
        # Ensure error message is ASCII-safe
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        logger.error(f"Training failed: {error_msg}")
        update_job_status(args.job_id, 'failed', error_msg)  # Use ASCII-safe error message
        sys.exit(1)

if __name__ == "__main__":
    main()
