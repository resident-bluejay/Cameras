from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from cameratutorial import wide_deep

tf.logging.set_verbosity(tf.logging.ERROR)

TEST_INPUT = ('Canon PowerShot A570 IS,2007,3072.0,2592.0,7.0,35.0,'
              '140.0,45.0,5.0,16.0,215.0,90.0,<=1000')
TEST_INPUT_VALUES = {
    'model': 'Canon PowerShot A570 IS',
    'release_date': 2007,
    'max_resolution': 3072.0,
    'low_resolution': 2592.0,
    'effective_pixels': 7.0,
    'zoom_wide': 35.0,
    'zoom_tele': 140.0,
    'normal_focus': 45.0,
    'macro_focus': 5.0,
    'storage_included':16.0,
    'wieght': 215.0,
    'dimension': 90.0,
    'price': '<=1000'
}

TEST_CSV = os.path.join(os.path.dirname(__file__), 'edited_dataset_test.csv')


class BaseTest(tf.test.TestCase):

  def setUp(self):
    # Create temporary CSV file
    self.temp_dir = self.get_temp_dir()
    self.input_csv = os.path.join(self.temp_dir, 'test.csv')
    with tf.gfile.Open(self.input_csv, 'w') as temp_csv:
      temp_csv.write(TEST_INPUT)

  def test_input_fn(self):
    dataset = wide_deep.input_fn(self.input_csv, 1, False, 1)
    features, labels = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      features, labels = sess.run((features, labels))

      # Compare the two features dictionaries.
      for key in TEST_INPUT_VALUES:
        self.assertTrue(key in features)
        self.assertEqual(len(features[key]), 1)
        feature_value = features[key][0]

        # Convert from bytes to string for Python 3.
        if isinstance(feature_value, bytes):
          feature_value = feature_value.decode()

        self.assertEqual(TEST_INPUT_VALUES[key], feature_value)

      self.assertFalse(labels)

  def build_and_test_estimator(self, model_type):
    """Ensure that model trains and minimizes loss."""
    model = wide_deep.build_estimator(self.temp_dir, model_type)

    # Train for 1 step to initialize model and evaluate initial loss
    model.train(
        input_fn=lambda: wide_deep.input_fn(
            TEST_CSV, num_epochs=1, shuffle=True, batch_size=1),
        steps=1)
    initial_results = model.evaluate(
        input_fn=lambda: wide_deep.input_fn(
            TEST_CSV, num_epochs=1, shuffle=False, batch_size=1))

    # Train for 100 epochs at batch size 3 and evaluate final loss
    model.train(
        input_fn=lambda: wide_deep.input_fn(
            TEST_CSV, num_epochs=100, shuffle=True, batch_size=3))
    final_results = model.evaluate(
        input_fn=lambda: wide_deep.input_fn(
            TEST_CSV, num_epochs=1, shuffle=False, batch_size=1))

    print('%s initial results:' % model_type, initial_results)
    print('%s final results:' % model_type, final_results)

    # Ensure loss has decreased, while accuracy and both AUCs have increased.
    self.assertLess(final_results['loss'], initial_results['loss'])
    self.assertGreater(final_results['auc'], initial_results['auc'])
    self.assertGreater(final_results['auc_precision_recall'],
                       initial_results['auc_precision_recall'])
    self.assertGreater(final_results['accuracy'], initial_results['accuracy'])

  def test_wide_deep_estimator_training(self):
    self.build_and_test_estimator('wide_deep')


if __name__ == '__main__':
  tf.test.main()
