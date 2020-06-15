Using Attacks
=============

Interface for Attack
--------------------

Please read documentation for :doc:`../api/realsafe.attack`.

There are two kinds of attacks:

- Attacks that supports attacking multiple images simultaneously (e.g. all white-box attacks). Their abstract base class is :class:`BatchAttack <realsafe.attack.base.BatchAttack>`. They have a ``batch_attack(xs)`` method.
- Attacks that attacks images one by one (e.g. NES). Their abstract base class is :class:`Attack <realsafe.attack.base.Attack>`. They have a ``attack(x)`` method.

Using Attacks
-------------

An example for using :class:`MIM <realsafe.attack.mim.MIM>`:

.. code-block:: python

   loss = CrossEntropyLoss(model)
   attack = MIM(
       model=model,
       batch_size=batch_size,
       loss=loss,
       goal='ut',
       distance_metric='l_inf',
       session=session,
   )
   attack.config(
       iteration=10,
       decay_factor=1.0,
       magnitude=8.0 / 255.0,
       alpha=1.0 / 255.0,
   )
   xs_adv = attack.batch_attack(xs, ys=ys)

We guarantee the ``config()`` method is cheap and support partial configuration (and later config for same parameter would override its old value).

.. note::

   When missing parameters, the ``batch_attack()`` or ``attack()`` method might throw errors saying something is ``None`` or TensorFlow complaining about uninitialized variables. These error messages are not clear now, though future improvements are possible. Please read the attack method's documentation before using it and make sure you have configured all needed parameters.
