import os


class Registry:
    '''The class for registry of modules.'''
    mapping = {
        "attacks": {},
        "models": {},
        "lr_schedulers": {},
        "transforms": {},
        "paths": {},
    }

    @classmethod
    def register_attack(cls, name=None, force=False):
        r"""Register an attack method to registry with key 'name'

        Args:
            name (str): Key with which the attacker will be registered.
            force (bool): Whether to register when the name has already existed in registry.
        """

        def wrap(attack):
            registerd_name = attack.__name__ if name is None else name
            if registerd_name in cls.mapping["attacks"] and not force:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        registerd_name, cls.mapping["attacks"][registerd_name]
                    )
                )

            cls.mapping["attacks"][registerd_name] = attack
            return attack

        return wrap

    @classmethod
    def register_model(cls, name=None, force=False):
        r"""Register a model to registry with key 'name'

        Args:
            name (str): Key with which the attacker will be registered.
            force (bool): Whether to register when the name has already existed in registry.
        """

        def wrap(model):
            registerd_name = model.__name__ if name is None else name
            if registerd_name in cls.mapping["models"] and not force:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        registerd_name, cls.mapping["models"][registerd_name]
                    )
                )
            cls.mapping["models"][registerd_name] = model
            return model

        return wrap

    @classmethod
    def register_lr_scheduler(cls, name=None, force=False):
        r"""Register a learning rate scheduler to registry with key 'name'

        Args:
            name (str): Key with which the attacker will be registered.
            force (bool): Whether to register when the name has already existed in registry.
        """

        def wrap(lr_scheduler):
            registerd_name = lr_scheduler.__name__ if name is None else name
            if registerd_name in cls.mapping["lr_schedulers"] and not force:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        registerd_name, cls.mapping["lr_schedulers"][registerd_name]
                    )
                )

            cls.mapping["lr_schedulers"][registerd_name] = lr_scheduler
            return lr_scheduler

        return wrap

    @classmethod
    def register_transform(cls, name=None, force=False):
        r"""Register a transform to registry with key 'name'

        Args:
            name (str): Key with which the attacker will be registered.
            force (bool): Whether to register when the name has already existed in registry.
        """

        def wrap(transform):
            registerd_name = transform.__name__ if name is None else name
            if registerd_name in cls.mapping["transforms"] and not force:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        registerd_name, cls.mapping["transforms"][registerd_name]
                    )
                )

            cls.mapping["transforms"][registerd_name] = transform
            return transform

        return wrap

    @classmethod
    def register_path(cls, name, path):
        r"""Register a path to registry with key 'name'

        Args:
            name (str): Key with which the path will be registered.
        """
        assert isinstance(path, str), "All path must be str."
        if name in cls.mapping["paths"]:
            return
        if not os.path.exists(path):
            os.makedirs(path)
        cls.mapping["paths"][name] = path

    @classmethod
    def get_attack(cls, name):
        '''Get a attack method by given name.'''
        if cls.mapping["attacks"].get(name, None):
            return cls.mapping["attacks"].get(name)
        raise KeyError(f'{name} is not registered!')

    @classmethod
    def get_model(cls, name):
        '''Get a model object by given name.'''
        if cls.mapping["models"].get(name, None):
            return cls.mapping["models"].get(name)
        raise KeyError(f'{name} is not registered!')

    @classmethod
    def get_lr_scheduler(cls, name):
        '''Get a lr scheduler object by given name.'''
        if cls.mapping["lr_schedulers"].get(name, None):
            return cls.mapping["lr_schedulers"].get(name)
        raise KeyError(f'{name} is not registered!')

    @classmethod
    def get_transform(cls, name):
        '''Get a transform object by given name.'''
        if cls.mapping["transforms"].get(name, None):
            return cls.mapping["transforms"].get(name)
        raise KeyError(f'{name} is not registered!')

    @classmethod
    def get_path(cls, name):
        '''Get a path by given name.'''
        if cls.mapping["paths"].get(name, None):
            return cls.mapping["paths"].get(name)
        raise KeyError(f'{name} is not registered!')

    @classmethod
    def list_attacks(cls):
        '''List all attack methods registered.'''
        return sorted(cls.mapping["attacks"].keys())

    @classmethod
    def list_models(cls):
        '''List all model classes registered.'''
        return sorted(cls.mapping["models"].keys())

    @classmethod
    def list_lr_schedulers(cls):
        '''List all lr schedulers registered.'''
        return sorted(cls.mapping["models"].keys())

    @classmethod
    def list_transforms(cls):
        '''List all transforms registered.'''
        return sorted(cls.mapping["models"].keys())


registry = Registry()
