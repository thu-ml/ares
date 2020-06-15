Running Benchmarks
==================

Using Python Interface
----------------------

Please read documentation for :doc:`../api/realsafe.benchmark`.

The workflow would be like:

.. code-block:: python

   # import whatever benchmark you want
   from realsafe.benchmark.distortion import DistortionBenchmark
   from realsafe.model.loader import load_model_from_path
   from realsafe.dataset import cifar10

   session = ...  # load tf.Session
   model = load_model_from_path('path/to/the/model.py').load(session)
   dataset = cifar10.load_dataset_for_classifier(model, load_target=True)

   # read documentation for the benchmark for all parameters
   benchmark = DistortionBenchmark(attack_name='mim', model=model, ...)
   # config the attack method
   benchmark.config(decay_factor=1.0)

   result = benchmark.run(dataset, some_logger)

Using Command Line Interface
----------------------------

Besides the Python interface, we also provide a much more convenient command line interface. All modules under ``realsafe.benchmark`` end in ``_cli`` could be called directly in command line:

.. code-block:: shell

   python3 -m realsafe.benchmark.iteration_cli --help
   python3 -m realsafe.benchmark.distortion_cli --help

You need to include options for the benchmark (e.g. ``--distortion``), and options for the attack method. Both attacks' ``__init__()``'s parameters and ``config()``'s parameters are needed. For example, for BIM, you need to include the ``--decay-factor`` option. The error message when missing parameters is not friendly now. Please check the documentation of the attack method's ``__init__()`` and ``config()`` for their parameters.

The ``test/test_distortion.sh`` and ``test/test_iteration.sh`` include many examples.
