"""CPU-only mathematical and one-update regression tests."""
import importlib
import unittest
import torch
try:
    grpo = importlib.import_module('claude08_grpo')
except ModuleNotFoundError as exc:
    if exc.name != 'claude08_grpo': raise
    grpo = None

class GRPOTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(grpo, 'Review08 pure Torch GRPO implementation is required')

    def test_population_standard_deviation_and_zero_group(self):
        values = torch.tensor([[0., 1., 2., 3.], [7., 7., 7., 7.]])
        got = grpo.group_advantages(values)
        torch.testing.assert_close(got[0], (values[0] - 1.5) / values[0].std(unbiased=False).clamp_min(1e-8))
        self.assertTrue(torch.equal(got[1], torch.zeros(4)))
        with self.assertRaises(ValueError): grpo.group_advantages(torch.ones(2, 3))

    def test_causal_shift_and_excluded_rows_have_zero_gradient(self):
        logits = torch.zeros(1, 5, 7, requires_grad=True)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        mask = torch.tensor([[False, False, False, True, True]])
        new = grpo.causal_logprobs(logits, ids)[0]
        old = new.detach().clone().requires_grad_(True)
        ref = new.detach().clone().requires_grad_(True)
        loss, stats = grpo.episode_loss([new], [old], [ref], [mask[0, 1:]], torch.tensor([1.]))
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0, :2], torch.zeros(2, 7)))
        self.assertGreater(float(logits.grad[0, 2:4].abs().sum()), 0)
        self.assertTrue(torch.equal(logits.grad[0, 4], torch.zeros(7)))
        self.assertIsNone(old.grad); self.assertIsNone(ref.grad)
        self.assertEqual(stats['action_tokens'], 2)

    def test_advantage_gradient_direction_and_equal_episode_weight(self):
        for advantage in [1., -1.]:
            value = torch.tensor([0.], requires_grad=True)
            loss, _ = grpo.episode_loss([value], [torch.zeros(1)], [torch.zeros(1)],
                                       [torch.ones(1, dtype=torch.bool)], torch.tensor([advantage]))
            loss.backward()
            self.assertLess(float(value.grad) * advantage, 0)
        a = torch.zeros(1, requires_grad=True); b = torch.zeros(9, requires_grad=True)
        loss, _ = grpo.episode_loss([a, b], [a.detach(), b.detach()], [a.detach(), b.detach()],
            [torch.ones(1, dtype=torch.bool), torch.ones(9, dtype=torch.bool)], torch.ones(2))
        loss.backward()
        self.assertAlmostEqual(float(a.grad.sum()), float(b.grad.sum()), places=6)

    def test_clipping_kl_formula_and_nonfinite_stop(self):
        new = torch.tensor([.5, -.5], requires_grad=True)
        old = torch.zeros(2); ref = torch.tensor([.1, .2])
        loss, _ = grpo.episode_loss([new], [old], [ref], [torch.ones(2, dtype=torch.bool)], torch.tensor([1.]))
        expected = (-torch.minimum(new.exp(), new.exp().clamp(.8, 1.2)) + .03 * ((ref-new).exp()-(ref-new)-1)).mean()
        torch.testing.assert_close(loss, expected)
        with self.assertRaises(FloatingPointError):
            grpo.episode_loss([torch.tensor([float('nan')])], [torch.zeros(1)], [torch.zeros(1)],
                              [torch.ones(1, dtype=torch.bool)], torch.ones(1))

    def sample(self):
        return grpo.ActionSample((1,2,3), (False,True,True), (-1.,-1.), (-1.,-1.))

    def test_records_immutable_and_hash_includes_masks(self):
        sample = self.sample(); before = sample.sha256
        self.assertEqual(sample.sha256, before)
        with self.assertRaises((AttributeError, TypeError)): sample.old_logprobs = (0.,0.)
        changed = grpo.ActionSample((1,2,3), (False,False,True), (-1.,-1.), (-1.,-1.))
        self.assertNotEqual(before, changed.sha256)

    def test_one_update_changes_parameters_and_preserves_old_reference(self):
        p = torch.nn.Parameter(torch.tensor(-1.))
        samples = [[self.sample()] for _ in range(4)]
        hashes = [s[0].sha256 for s in samples]
        def score(sample):
            return p.expand(2) if sample is samples[3][0] else p.expand(2) * .5 - .5
        result = grpo.one_update([p], samples, [[0.,1.,2.,3.]], score)
        self.assertEqual(result['optimizer_updates'], 1)
        self.assertTrue(result['finite_gradients'])
        self.assertGreater(result['parameter_delta_l2'], 0)
        self.assertEqual(hashes, [s[0].sha256 for s in samples])
        old = p.detach().clone()
        result = grpo.one_update([p], samples, [[2.,2.,2.,2.]], lambda _: self.fail('zero group scored'))
        self.assertEqual(result['status'], 'NO_UPDATE_ZERO_GROUP_VARIANCE')
        self.assertTrue(torch.equal(p.detach(), old))

    def test_std_at_below_and_above_epsilon(self):
        for scale in [.5e-8,1e-8,2e-8]:
            returns=torch.tensor([[-scale,-scale,scale,scale]],dtype=torch.float64)
            got=grpo.group_advantages(returns)
            if scale<=1e-8:
                self.assertTrue(torch.equal(got,torch.zeros_like(got)))
                p=torch.nn.Parameter(torch.tensor(-1.))
                result=grpo.one_update([p],[[self.sample()] for _ in range(4)],returns,
                                       lambda _:self.fail('tiny-variance group performed a forward'))
                self.assertEqual(result['status'],'NO_UPDATE_ZERO_GROUP_VARIANCE')
                self.assertEqual(float(p.detach()),-1.)
            else:
                torch.testing.assert_close(got,torch.tensor([[-1.,-1.,1.,1.]],dtype=torch.float64))

    def test_sequential_backward_matches_full_episode_gradient(self):
        p=torch.nn.Parameter(torch.tensor(-1.))
        short=grpo.ActionSample((1,2),(False,True),(-1.,),(-1.,))
        long=self.sample()
        episodes=[[short,long] for _ in range(4)]
        returns=[[0.,1.,2.,3.]]
        adv=grpo.group_advantages(returns).flatten()
        factors=[.05,.1,.15,.2]
        full=[]
        for factor in factors:
            full.append(torch.cat([p.expand(1)*factor+factor-1,p.expand(2)*(factor+.05)+(factor+.05)-1]))
        loss,_=grpo.episode_loss(full,[torch.full((3,),-1.) for _ in range(4)],
             [torch.full((3,),-1.) for _ in range(4)],[torch.ones(3,dtype=torch.bool) for _ in range(4)],adv)
        expected=torch.autograd.grad(loss,p)[0]
        calls=0; captured=[]
        def score(sample):
            nonlocal calls
            if calls>0: self.assertIsNotNone(p.grad,'Previous action must backpropagate before scoring the next')
            factor=factors[calls//2]+(.05 if calls%2 else 0)
            calls+=1
            return p.expand(len(sample.input_ids)-1)*factor+factor-1
        grpo.one_update([p],episodes,returns,score,before_step=lambda:captured.append(p.grad.detach().clone()))
        torch.testing.assert_close(captured[0],expected)

if __name__ == '__main__': unittest.main()
