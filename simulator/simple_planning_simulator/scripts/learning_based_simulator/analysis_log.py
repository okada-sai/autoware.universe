#!/usr/bin/env python3

import argparse

from learning_based_simulator import LearningBasedSimulator

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-l', '--log-file', default='log.csv', help='log_file name')
  parser.add_argument('-m', '--model-file', default='model.pt', help='model_file name')
  parser.add_argument('-v', '--value', default='acc', help='value name to analysis (acc, vel, steer, ...)')
  args = parser.parse_args()

  sim = LearningBasedSimulator(model_path='model/' + args.model_file)
  sim.analysis_log(log_path='log/' + args.log_file, value=args.value)
