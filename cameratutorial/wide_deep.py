from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys

import tensorflow as tf
_CSV_COLUMNS = [
    'model', 'release_date', 'max_resolution', 'low_resolution', 'effective_pixels',
    'zoom_wide', 'zoom_tele', 'normal_focus_range', 'macro_focus_range', 'storage_included',
    'weight', 'dimensions', 'price'
]

_CSV_COLUMN_DEFAULTS = [[''], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], ['']]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='/camera_data',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

parser.add_argument(
   # '--train_data', type=str, default='/camera_data/camera_dataset.csv',
    '--train_data', type=str, default='/camera_data/edited_dataset.csv',
    help='Path to the training data.')

parser.add_argument(
    #'--test_data', type=str, default='/camera_data/camera_dataset.csv',
    '--test_data', type=str, default='/camera_data/edited_dataset_test.csv',
    help='Path to the test data.')

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  release_date = tf.feature_column.numeric_column('release_date')
  max_resolution = tf.feature_column.numeric_column('max_resolution')
  low_resolution = tf.feature_column.numeric_column('low_resolution')
  effective_pixels = tf.feature_column.numeric_column('effective_pixels')
  zoom_wide = tf.feature_column.numeric_column('zoom_wide')
  zoom_tele = tf.feature_column.numeric_column('zoom_tele')
  normal_focus_range = tf.feature_column.numeric_column('normal_focus-range')
  macro_focus_range = tf.feature_column.numeric_column('macro_focus_range')
  storage_included = tf.feature_column.numeric_column('storage_included')
  weight = tf.feature_column.numeric_column('weight')
  dimensions = tf.feature_column.numeric_column('dimensions')
  #price = tf.feature_column.numeric_column('price')

  model = tf.feature_column.categorical_column_with_vocabulary_list(
      'model', [
          'Afga', 'Canon', 'Casio', 'Epson', 'Fujifilm', 'HP Photosmart',
          'JVC', 'Kodak', 'Kyocera', 'Leica', 'Nikon',
          'Olympus', 'Panasonic', 'Pentax', 'Ricoh', 'Samsung', 'Sanyo'
          'Sigma', 'Sony', 'Toshiba'
      ])

  # Transformations.
  year_buckets = tf.feature_column.bucketized_column(
      release_date, boundaries=[1990, 1995, 2000, 2005, 2010, 2015])

  # Wide columns and deep columns.
  base_columns = [

      year_buckets, max_resolution, effective_pixels, storage_included, weight,
      dimensions, zoom_tele, macro_focus_range
  ]

  crossed_columns = [
      tf.feature_column.crossed_column(
          ['weight', 'dimensions'], hash_bucket_size=100),
      tf.feature_column.crossed_column(
          ['max_resolution', 'effective_pixels'], hash_bucket_size= 100 ),
      tf.feature_column.crossed_column(
          ['zoom_tele', 'macro_focus_range'], hash_bucket_size = 100 ),
      ]

  wide_columns = base_columns + crossed_columns

  deep_columns = [ release_date,
                   max_resolution,
                   effective_pixels,
                   storage_included,
                   weight,
                   dimensions,
                   zoom_tele,
                   macro_focus_range
                   ]

  return wide_columns, deep_columns

def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have either run data_download.py or '
        'set both arguments --train_data and --test_data.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))

        labels = features.pop('price')
        return features, tf.equal(labels, '>1000')

        # Extract lines from input files using the Dataset API.

    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def main(unused_argv):
    # Clean up the model directory if present
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(
            FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

        results = model.evaluate(input_fn=lambda: input_fn(
            FLAGS.test_data, 1, False, FLAGS.batch_size))

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

