"""Regression tests for optional transformer positional embeddings.

Run from the repository root: python -m unittest discover -s tests -v
Requires PyTorch.
"""
import unittest

import torch

from model.Defor_transformer import DeformableTransformer


class DeformableTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = DeformableTransformer(dim=8)
        self.query = torch.randn(2, 3, 8)
        self.value = torch.randn(2, 16, 8)

    def test_omitted_and_none_positions_match_zero_positions(self):
        expected = self.model(self.query, self.value, torch.zeros_like(self.query))
        for kwargs in ({}, {"pos": None}):
            with self.subTest(kwargs=kwargs):
                actual = self.model(self.query, self.value, **kwargs)
                self.assertEqual(actual.shape, (2, 3, 8))
                torch.testing.assert_close(actual, expected)

    def test_nonzero_positions_are_added(self):
        pos = torch.randn_like(self.query)
        expected = self.model.encoder1(self.query + pos, self.value)
        actual = self.model(self.query, self.value, pos)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, self.model.encoder1(self.query, self.value)))

    def test_backward_without_positions(self):
        query = self.query.clone().requires_grad_()
        value = self.value.clone().requires_grad_()
        self.model(query, value).square().sum().backward()
        for name, tensor in (("query", query), ("value", value)):
            with self.subTest(input=name):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all())
                self.assertGreater(tensor.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
