#!/usr/bin/env python
""" Attribution to https://competitions.codalab.org/competitions/18405#participate
we adopted the evaluation pipeline to our needs.

You can find more detailed explanations of the metrics here: https://www.researchgate.net/publication/286764719_Leaf_segmentation_in_plant_phenotyping_a_collation_study
"""

import argparse
import os
import os.path
import sys

import h5py
import numpy as np


def evaluate(inLabelFileName, gtLabelFileName):

  inLabelFile = h5py.File(inLabelFileName, 'r')
  gtLabelFile = h5py.File(gtLabelFileName, 'r')

  # loop all datasets in gtLabelFile and compare to respective dataset in inLabelFile
  results = {
      'header': ('number', 'filename', 'SymBestDice', 'FgBgDice', 'aDiC', 'DiC',
                 'Pa', 'Pa±1')
  }
  stats = {'header': ('measure', 'mean', 'std')}
  for key in gtLabelFile.keys():  # loop A1 to A5
    group = gtLabelFile.get(key)
    SymBestDice = []
    FgBgDice = []
    absDiC = []
    DiC = []
    Pa = []
    PaPM1 = []
    filenames = []
    imgNum = 0
    for data in group.keys():  # loop datasets in the current group
      # get label image
      fullkey = key + '/' + data + '/label'

      print(fullkey)

      gtLabel = np.asarray(gtLabelFile.get(fullkey))
      if fullkey in inLabelFile:
        inLabel = np.asarray(inLabelFile.get(fullkey))
      else:
        print(
            "warning - did not find any associated predictions for current gt image"
        )
        inLabel = np.zeros(gtLabel.shape)

      # now call the scoring routines and append results to respective lists
      SymBestDice.append(
          np.minimum(BestDice(inLabel, gtLabel), BestDice(gtLabel, inLabel)))
      FgBgDice.append(FGBGDice(inLabel, gtLabel))
      absDiC.append(absDiffCount(inLabel, gtLabel))
      DiC.append(DiffCount(inLabel, gtLabel))
      Pa.append(PercentageAgree(inLabel, gtLabel))
      PaPM1.append(PercentageAgreePM1(inLabel, gtLabel))

      imgNum += 1

      #  get the original filename
      fullkey = key + '/' + data + '/label_filename'
      filename = str(np.asarray(gtLabelFile.get(fullkey)))
      filenames.append(filename)

    # store results in dictionary
    zipped = zip(
        range(1, imgNum + 1), filenames, SymBestDice, FgBgDice, absDiC, DiC, Pa, PaPM1)
    groupresults = {key: list(zipped)}
    groupstats = {
        key: [('SymBestDice', np.mean(SymBestDice), np.std(SymBestDice)),
              ('FgBgDice', np.mean(FgBgDice), np.std(FgBgDice)),
              ('aDiC', np.mean(absDiC), np.std(absDiC)),
              ('DiC', np.mean(DiC), np.std(DiC)),
              ('Pa', np.mean(Pa), np.std(Pa)),
              ('Pa±1', np.mean(PaPM1), np.std(PaPM1))]
    }
    results.update(groupresults)
    stats.update(groupstats)

  gtLabelFile.close()
  inLabelFile.close()

  return results, stats


##############################################################################
def absDiffCount(inLabel, gtLabel):
  # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
  #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
  # output: absolute difference in count

  # check if label images have same size
  assert (inLabel.shape == gtLabel.shape), "Dimensions do not match"

  # number of leaves in prediction
  n_pred_leaves = float(np.sum(np.unique(inLabel) > 0))

  # number of leaves in ground turth
  n_gt_leaves = float(np.sum(np.unique(gtLabel) > 0))

  adic = abs(n_pred_leaves - n_gt_leaves)

  return adic


##############################################################################
def DiffCount(inLabel, gtLabel):
  # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
  #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
  # output: difference in count

  # check if label images have same size
  assert (inLabel.shape == gtLabel.shape), "Dimensions do not match"

  # number of leaves in prediction
  n_pred_leaves = float(np.sum(np.unique(inLabel) > 0))

  # number of leaves in ground turth
  n_gt_leaves = float(np.sum(np.unique(gtLabel) > 0))

  dic = n_pred_leaves - n_gt_leaves

  return dic


##############################################################################
def PercentageAgree(inLabel, gtLabel):
  # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
  #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
  # output: Agreement between ground truth and prediction

  # check if label images have same size
  assert (inLabel.shape == gtLabel.shape), "Dimensions do not match"

  # number of leaves in prediction
  n_pred_leaves = float(np.sum(np.unique(inLabel) > 0))

  # number of leaves in ground turth
  n_gt_leaves = float(np.sum(np.unique(gtLabel) > 0))

  agree = float(n_pred_leaves == n_gt_leaves)

  return agree


##############################################################################
def PercentageAgreePM1(inLabel, gtLabel):
  # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
  #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
  # output: Agreement between ground truth and prediction, but does allow for an offset of one leave

  # check if label images have same size
  assert (inLabel.shape == gtLabel.shape), "Dimensions do not match"

  # number of leaves in prediction
  n_pred_leaves = float(np.sum(np.unique(inLabel) > 0))

  # number of leaves in ground turth
  n_gt_leaves = float(np.sum(np.unique(gtLabel) > 0))

  diff = np.abs(n_pred_leaves - n_gt_leaves)

  agree = float(diff <= 1)

  return agree


##############################################################################
def BestDice(inLabel, gtLabel):
  # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
  #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
  # output: score: Dice score
  #
  # We assume that the lowest label in inLabel is background, same for gtLabel
  # and do not use it. This is necessary to avoid that the trivial solution,
  # i.e. finding only background, gives excellent results.
  #
  # For the original Dice score, labels corresponding to each other need to
  # be known in advance. Here we simply take the best matching label from
  # gtLabel in each comparison. We do not make sure that a label from gtLabel
  # is used only once. Better measures may exist. Please enlighten me if I do
  # something stupid here...

  score = 0  # initialize output

  # check if label images have same size
  assert (inLabel.shape == gtLabel.shape), "Dimensions do not match"

  maxInLabel = np.max(inLabel)  # maximum label value in inLabel
  minInLabel = np.min(inLabel)  # minimum label value in inLabel
  InLabelIDs = np.unique(inLabel)
  if 0 in InLabelIDs:
    InLabelIDs = InLabelIDs[1:]  # remove background

  maxGtLabel = np.max(gtLabel)  # maximum label value in gtLabel
  minGtLabel = np.min(gtLabel)  # minimum label value in gtLabel
  GtLabelIDs = np.unique(gtLabel)
  if 0 in GtLabelIDs:
    GtLabelIDs = GtLabelIDs[1:]  # remove background

  if (maxInLabel == minInLabel):  # trivial solution (only background)
    return score

  for i in InLabelIDs:  # loop all labels of inLabel, but background
    sMax = 0
    # maximum Dice value found for label i so far
    for j in GtLabelIDs:  # loop all labels of gtLabel, but background
      s = Dice(inLabel, gtLabel, i, j)  # compare labelled regions
      # keep max Dice value for label i
      if (sMax < s):
        sMax = s
    score = score + sMax
    # sum up best found values
  score = score / len(InLabelIDs)
  return score


##############################################################################
def FGBGDice(inLabel, gtLabel):
  # input: inLabel: label image to be evaluated. Background label is assumed to be the lowest one.
  #        gtLabel: ground truth label image. Background label is assumed to be the lowest one.
  # output: Dice score for foreground/background segmentation, only.

  # check if label images have same size
  assert (inLabel.shape == gtLabel.shape), "Dimensions do not match"

  minInLabel = np.min(inLabel)  # minimum label value in inLabel := background
  minGtLabel = np.min(gtLabel)  # minimum label value in gtLabel := background

  one = np.ones(inLabel.shape)
  inFgLabel = (inLabel != minInLabel * one) * one
  gtFgLabel = (gtLabel != minGtLabel * one) * one

  return Dice(inFgLabel, gtFgLabel, 1, 1)  # Dice score for the foreground


##############################################################################
def Dice(inLabel, gtLabel, i, j):
  # calculate Dice score for the given labels i and j

  # check if label images have same size
  assert (inLabel.shape == gtLabel.shape), "Dimensions do not match"

  one = np.ones(inLabel.shape)
  inMask = (inLabel == i * one)  # find region of label i in inLabel
  gtMask = (gtLabel == j * one)  # find region of label j in gtLabel
  inSize = np.sum(inMask * one)  # cardinality of set i in inLabel
  gtSize = np.sum(gtMask * one)  # cardinality of set j in gtLabel
  overlap = np.sum(inMask * gtMask *
                   one)  # cardinality of overlap of the two regions
  if ((inSize + gtSize) > 1e-8):
    out = 2 * overlap / (inSize + gtSize)  # Dice score
  else:
    out = 0

  return out


##############################################################################
def WriteOutput(output_filename, somedict):
  # output_filename: name of the output file
  # results: array containing the result values
  with open(output_filename, 'w') as output_file:
    # write header if available
    if 'header' in somedict:
      output_file.write('dataset,')
      output_file.write(','.join(map(str, somedict['header'])))
      output_file.write('\n')
    # write rest
    for key in somedict:
      # skip header
      if key.find('header') == -1:
        # get every line and output it
        for line in somedict[key]:
          output_file.write(key)
          output_file.write(',')
          output_file.write(','.join(map(str, line)))
          output_file.write('\n')

def is_h5_file(filename: str) -> bool:
  """ Check if given file is an hdf5 file.

  Args:
      filename (str): filename

  Returns:
      bool: whether or not given file is an hdf5 file
  """
  if filename.endswith(".h5"):
    return True
  else:
    return False


##############################################################################
# main routine
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--pred",
      required=True,
      help='path to .h5 file which contains all predictions')
  parser.add_argument(
      '--gt',
      required=True,
      help='path to .h5 file which contains ground truth')
  parser.add_argument("--out", required=True, help='path to output directory')
  args = vars(parser.parse_args())

  valid_pred_file = is_h5_file(args['pred'])
  valid_gt_file = is_h5_file(args['gt'])

  if not os.path.isdir(args['out']):
    os.makedirs(args['out'])

  if valid_pred_file and valid_gt_file:
    # call the evaluation routine dealing with these files
    results, stats = evaluate(args['pred'], args['gt'])

    # write results to output file
    output_filename = os.path.join(args['out'], 'details.txt')
    WriteOutput(output_filename, results)
    output_filename = os.path.join(args['out'], 'scores.txt')
    WriteOutput(output_filename, stats)
