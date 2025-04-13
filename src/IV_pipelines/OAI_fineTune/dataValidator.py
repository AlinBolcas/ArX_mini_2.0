import sys
from pathlib import Path
import json
import tiktoken
import numpy as np
from collections import defaultdict
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities
from modules.VIII_utils.utils import printColoured

# Custom formatter for prettier logging
class PrettyFormatter(logging.Formatter):
    def format(self, record):
        if 'STEP' in record.msg:
            return f"\n{printColoured('='*50, 'grey')}\n{printColoured(record.msg, 'cyan')}\n{printColoured('='*50, 'grey')}"
        elif 'Error' in record.msg:
            return f"\n{printColoured('❌ ' + record.msg, 'red')}"
        elif 'Success' in record.msg:
            return f"\n{printColoured('✓ ' + record.msg, 'green')}"
        elif 'Warning' in record.msg:
            return f"\n{printColoured('⚠️ ' + record.msg, 'yellow')}"
        else:
            return f"{printColoured('>', 'blue')} {record.msg}"

# Setup logging
handler = logging.StreamHandler()
handler.setFormatter(PrettyFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.MAX_TOKENS_PER_EXAMPLE = 16385
        self.TARGET_EPOCHS = 3
        self.MIN_TARGET_EXAMPLES = 100
        self.MAX_TARGET_EXAMPLES = 25000
    
    def load_jsonl(self, file_path: str) -> list:
        """Load and validate JSONL dataset"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dataset = [json.loads(line) for line in f]
            logger.info(f"Loaded {len(dataset)} examples from {file_path}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading JSONL: {e}")
            return []

    def markdown_to_jsonl(self, md_path: str, output_path: str) -> bool:
        """Convert markdown training data to JSONL format"""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split into sections by '---'
            sections = content.split('---\n')
            dataset = []

            for section in sections:
                if not section.strip():
                    continue
                
                # Parse sections
                parts = section.split('# ')
                message_map = {}
                current_key = None
                
                for part in parts:
                    if not part.strip():
                        continue
                    lines = part.strip().split('\n', 1)
                    if len(lines) == 2:
                        key, content = lines
                        message_map[key.strip()] = content.strip()

                if 'system' in message_map and 'user' in message_map and 'assistant' in message_map:
                    training_item = {
                        "messages": [
                            {"role": "system", "content": message_map['system']},
                            {"role": "user", "content": message_map['user']},
                            {"role": "assistant", "content": message_map['assistant']}
                        ]
                    }
                    dataset.append(training_item)

            # Save as JSONL
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item) + '\n')

            logger.info(f"Successfully converted {len(dataset)} examples to JSONL")
            return True

        except Exception as e:
            logger.error(f"Error converting markdown to JSONL: {e}")
            return False

    def check_format(self, dataset: list) -> dict:
        """Validate dataset format for OpenAI fine-tuning"""
        format_errors = defaultdict(int)

        for ex in dataset:
            if not isinstance(ex, dict):
                format_errors["data_type"] += 1
                continue
                
            messages = ex.get("messages", None)
            if not messages:
                format_errors["missing_messages_list"] += 1
                continue
                
            for message in messages:
                if "role" not in message or "content" not in message:
                    format_errors["message_missing_key"] += 1
                
                if message.get("role", None) not in ("system", "user", "assistant"):
                    format_errors["unrecognized_role"] += 1
                    
                if not message.get("content", None) or not isinstance(message.get("content"), str):
                    format_errors["missing_content"] += 1
            
            if not any(message.get("role", None) == "assistant" for message in messages):
                format_errors["example_missing_assistant_message"] += 1

        if format_errors:
            logger.error("Found format errors:")
            for k, v in format_errors.items():
                logger.error(f"{k}: {v}")
        else:
            logger.info("No format errors found")
            
        return format_errors

    def analyze_token_counts(self, dataset: list) -> tuple:
        """Analyze token distribution in the dataset"""
        n_missing_system = 0
        n_missing_user = 0
        convo_lens = []
        assistant_lens = []

        for ex in dataset:
            messages = ex["messages"]
            if not any(message["role"] == "system" for message in messages):
                n_missing_system += 1
            if not any(message["role"] == "user" for message in messages):
                n_missing_user += 1
                
            total_tokens = sum(len(self.encoding.encode(m["content"])) for m in messages)
            convo_lens.append(total_tokens)
            
            assistant_tokens = sum(
                len(self.encoding.encode(m["content"]))
                for m in messages if m["role"] == "assistant"
            )
            assistant_lens.append(assistant_tokens)

        # Log statistics
        logger.info(f"Missing system messages: {n_missing_system}")
        logger.info(f"Missing user messages: {n_missing_user}")
        logger.info(f"Token statistics:")
        logger.info(f"Total tokens - min: {min(convo_lens)}, max: {max(convo_lens)}, mean: {np.mean(convo_lens):.0f}")
        logger.info(f"Assistant tokens - min: {min(assistant_lens)}, max: {max(assistant_lens)}, mean: {np.mean(assistant_lens):.0f}")

        n_too_long = sum(l > self.MAX_TOKENS_PER_EXAMPLE for l in convo_lens)
        if n_too_long:
            logger.warning(f"{n_too_long} examples exceed {self.MAX_TOKENS_PER_EXAMPLE} tokens and will be truncated")

        return convo_lens, assistant_lens

    def estimate_cost(self, convo_lens: list, config_type: str = "minimal"):
        """
        Estimate fine-tuning cost for minimal or maximal configuration
        
        Args:
            convo_lens: List of conversation token lengths
            config_type: Either "minimal" or "maximal"
        """
        n_examples = len(convo_lens)
        n_tokens = sum(min(self.MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
        
        # Configuration presets
        configs = {
            "minimal": {
                'n_epochs': 3,
                'batch_size': 'auto',
                'learning_rate_multiplier': 'auto'
            },
            "maximal": {
                'n_epochs': 12,  # More epochs for better learning
                'batch_size': 4,  # Smaller batch size for better convergence
                'learning_rate_multiplier': 1.6  # Higher learning rate for deeper optimization
            }
        }
        
        params = configs[config_type]
        n_epochs = params['n_epochs']
        total_tokens = n_epochs * n_tokens
        
        # Cost calculation with epoch scaling
        BASE_COST_PER_1K = 0.008
        HIGHER_EPOCH_DISCOUNT = 0.95  # 5% discount for epochs beyond 3
        
        if n_epochs > 3:
            base_cost = (n_tokens * 3 / 1000) * BASE_COST_PER_1K
            additional_cost = (n_tokens * (n_epochs - 3) / 1000) * BASE_COST_PER_1K * HIGHER_EPOCH_DISCOUNT
            total_cost = base_cost + additional_cost
        else:
            total_cost = (total_tokens / 1000) * BASE_COST_PER_1K
        
        # Log configuration details
        logger.info(f"\n{config_type.title()} Configuration:")
        logger.info(f"Examples: {n_examples} | Epochs: {n_epochs} | Batch size: {params['batch_size']} | LR mult: {params['learning_rate_multiplier']}")
        logger.info(f"Total tokens: {total_tokens:,} | Estimated cost: ${total_cost:.2f}")
        logger.info(f"Training time: {(total_tokens / 150 / 3600):.1f} hours")  # 150 tokens/sec
        
        return total_cost

def main():
    logger.info("STEP 1: Initializing Data Validator")
    validator = DataValidator()
    
    # Paths
    jsonl_path = project_root / "output" / "fineTune" / "cold_email_training.jsonl"
    md_path = project_root / "output" / "fineTune" / "cold_email_training.md"
    
    logger.info("STEP 2: Loading Dataset")
    if "--from-md" in sys.argv:
        logger.info("Converting markdown to JSONL")
        if not validator.markdown_to_jsonl(str(md_path), str(jsonl_path)):
            return
    
    dataset = validator.load_jsonl(str(jsonl_path))
    if not dataset:
        return
        
    logger.info("STEP 3: Checking Format")
    format_errors = validator.check_format(dataset)
    if format_errors:
        return
        
    logger.info("STEP 4: Analyzing Token Distribution")
    convo_lens, assistant_lens = validator.analyze_token_counts(dataset)
    
    logger.info("STEP 5: Estimating Training Costs")
    min_cost = validator.estimate_cost(convo_lens, "minimal")
    max_cost = validator.estimate_cost(convo_lens, "maximal")
    
    logger.info("\nSummary:")
    logger.info(f"Minimal training cost: ${min_cost:.2f}")
    logger.info(f"Optimal training cost: ${max_cost:.2f}")
    logger.info(f"Recommendation: {'Minimal config (good for testing)' if max_cost > 50 else 'Maximal config (best results)'}")
    
    logger.info("\nValidation complete! 🎉")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
    except Exception as e:
        logger.error(f"Error during validation: {e}")
    finally:
        logger.info("Process complete")