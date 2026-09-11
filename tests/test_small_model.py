import io
import unittest
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


class SmallModelTests(unittest.TestCase):
    def test_small_model_forward_backward_and_checkpoint(self):
        torch.manual_seed(42)
        config = MiniMindConfig(hidden_size=384, num_hidden_layers=8,
                               use_moe=False, max_position_embeddings=32)
        model = MiniMindForCausalLM(config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 17_210_496)
        self.assertEqual(config.head_dim, 48)
        self.assertEqual(config.intermediate_size, 1216)
        self.assertIs(model.model.embed_tokens.weight, model.lm_head.weight)
        ids = torch.randint(3, config.vocab_size, (1, 8))
        result = model(ids, labels=ids)
        self.assertEqual(tuple(result.logits.shape), (1, 8, 6400))
        self.assertTrue(torch.isfinite(result.loss).item())
        result.loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        data = io.BytesIO()
        torch.save(model.state_dict(), data)
        data.seek(0)
        restored = MiniMindForCausalLM(config)
        restored.load_state_dict(torch.load(data, weights_only=True), strict=True)
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(ids).logits, restored(ids).logits,
                                       rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
