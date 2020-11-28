import torch
from configs import Config
from module.dataloaders import get_dataloader
from module.vocabulary import Vocab
from module.trainer import SumTrainer
from models.simple_sum import MetricSum


if __name__ == "__main__":
    opts = Config()
    vocab = Vocab(opts.vocab_file, opts.vocab_size)
    train = get_dataloader(opts.cnndm_train, vocab, opts, shuffle=True)
    val = get_dataloader(opts.cnndm_val, vocab, opts)
    test = get_dataloader(opts.cnndm_test, vocab, opts)
    

    model = MetricSum(opts)
    if opts.cuda: model.cuda()
    trainer = SumTrainer(train, val, test, opts)

    trainer.train(model)
