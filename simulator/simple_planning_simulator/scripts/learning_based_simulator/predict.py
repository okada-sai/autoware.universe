#!/usr/bin/env python3

import argparse

from learning_based_simulator import LearningBasedSimulator

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-l', '--log-file', default='log_for_predict.csv', help='log_file name')
  parser.add_argument('-m', '--model-file', default='model.pt', help='model_file name')
  parser.add_argument('-u', '--use-lstm', action='store_true', help='use lstm or not')
  args = parser.parse_args()

  sim = LearningBasedSimulator(model_path='model/' + args.model_file, use_lstm=args.use_lstm)
  sim.predict_offline(log_path='log/' + args.log_file)
