# VAE (2013)

- Problem: latent variable-based generation model
- Solution: propose a variational auto-encoder model with elbo loss and reparameterization tricks.
- Key method:
  - Variational lower bound: This is theoretical proof of why minimizing elbo loss is lower or equivalent to minimizing `p(x)`, and also elbo loss enables us to sample `z` from certain distribution.
  - Reparametrization tricks: Modification from sampling to adding noise for backpropagation purposes.




![image](https://user-images.githubusercontent.com/31476895/211249550-1f30432d-d9df-4f50-82aa-327c28a0ff54.png)



## Reference

```tex
@article{kingma2013auto,
  title={Auto-encoding variational bayes},
  author={Kingma, Diederik P and Welling, Max},
  journal={arXiv preprint arXiv:1312.6114},
  year={2013}
}
```

