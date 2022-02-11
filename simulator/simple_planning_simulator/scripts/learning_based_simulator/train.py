#!/usr/bin/env python3

import argparse

from learning_based_simulator import LearningBasedSimulator

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-l', '--log-file', default='log.csv', help='log_file name')
  parser.add_argument('-e', '--epoch', type=int, default=50, help='epoch size')
  parser.add_argument('-b', '--batch', type=int, default=1000, help='batch size')
  parser.add_argument('-t', '--test', type=float, default=0.1, help='test size (ratio)')
  parser.add_argument('-m', '--model-file', default='', help='model_file name')
  parser.add_argument('-o', '--output-model-file', default='', help='output_model_file name')
  args = parser.parse_args()

  model_path = '' if args.model_file == '' else 'model/' + args.model_file

  sim = LearningBasedSimulator(model_path=model_path)
  sim.train_offline(log_path='log/' + args.log_file, output_model_path=args.output_model_file, epoch=args.epoch, batch_size=args.batch, test_size=args.test)
