import pickle
import inspect
import uuid
import copy
import time
from parallel.utils import DummyFuture


class ParallelStudyPoolBoy(object):

    # for parallel execution
    def __init__(self, futures, future_keys, ps):

        self._futures = futures
        self._future_keys = future_keys
        self._ps = ps
        self._N_total = len(self._futures)
        self._N_failed = 0
        self._N_finished = 0
        self._N_remaining = self._N_total
        self._delta_finished = False
        self._t_start = None

    @property
    def ps(self):
        return self._ps

    def __bool__(self):
        return self._N_remaining > 0

    def _sleep(self, mtime):
        time.sleep(mtime)

    def wait_for_results(self, T_SLEEP_INTERVAL, path = None, verbose = True, intermediate_save = True):

        self._t_start = time.time()

        if path is None:
            path = 'results_temporary'

        while self:

            self.check(path, intermediate_save)

            if intermediate_save and self._delta_finished:
                # save intermediate results if new jobs have finished
                self._ps.save(path)
                self._delta_finished = False

            if verbose:
                print(" {} / {} futures have finished ({} failed) [runtime = {}s] <<<<".format(self._N_finished, self._N_total, self._N_failed, time.time() - self._t_start))


            if self._N_remaining > 0:
                self._sleep(T_SLEEP_INTERVAL)




    def check(self, path, intermediate_save):

        futures_to_remove = list()
        future_keys_to_remove = list()
        future_indeces_to_remove = list()
        

        for ii in range(len(self._futures)):

            # because zip causes some weird problems (?), iterate manually
            future = self._futures[ii]
            future_key = self._future_keys[ii]
           
            if future.done():

                if isinstance(future, DummyFuture):
                    future.compute()

                self._delta_finished = True
                self._N_finished += 1
                self._N_remaining -= 1
                try:
                    res = future.result()
                    self._ps.put_dictionary_with_key(res, future_key, accumulate=True)
                except Exception as exc:
                    self._N_failed += 1
                    self._ps.notify_about_error_from_key(future_key, exc)

                futures_to_remove.append(future)
                future_keys_to_remove.append(future_key)
                future_indeces_to_remove.append(ii)

                if isinstance(future, DummyFuture) and intermediate_save:
                    print(
                        "SEQUENTIAL: >>>>>> {} / {} futures have finished ({} failed) [runtime = {}s] <<<<".format(self._N_finished,
                                                                                                       self._N_total,
                                                                                                       self._N_failed,
                                                                                                       time.time() - self._t_start))
                    self._ps.save(path)
                    self._delta_finished = False

        assert len(futures_to_remove) == len(future_keys_to_remove) == len(future_indeces_to_remove)
        if futures_to_remove:
            for kk in sorted(future_indeces_to_remove, reverse=True):
                del self._futures[kk]
                del self._future_keys[kk]




class ParameterStudy(object):

    def __init__(self):

        self._parameters = dict()

        self._parameters_ordered = list()
        self._cases = list()
        self._data = dict()
        self._errors = dict()

        self.info = dict()
        self._id = str(uuid.uuid4())


    @property
    def parameters(self):
        return self._parameters

    @classmethod
    def FromTemplate(cls, ps):

        mps = cls()
        mps._parameters = copy.copy(ps._parameters)
        mps._parameters_ordered = copy.copy(ps._parameters_ordered)
        return mps

    @classmethod
    def FromParameterStudies(cls, studies, accumulate = True):

        mstudy = ParameterStudy.FromTemplate(studies[0])

        for study in studies:
            mstudy.merge(study, accumulate=accumulate)

        return mstudy

    def summarize_errors(self):

        num_errors = sum([len(errs) for errs in self._errors.values()])
        print(">>> {} errors occured during computation <<< ".format(num_errors))

        for key, error in self._errors.items():
            print("Key:  {} || {}".format(tuple(key), error))


    def register_parameter(self, param, type, finalize = False):

        if not (inspect.isclass(type) or isinstance(type, list)):
            raise TypeError

        if param in self._parameters:
            if type != self._parameters[param]:
                raise RuntimeError

        self._parameters[param] = type
        self._parameters_ordered.append(param)

    def notify_about_error_from_key(self, gkey, exception):

        if gkey not in self._errors:
            self._errors[gkey] = list()
        self._errors[gkey].append(exception)


    def merge(self, ps, accumulate = True):

        assert self.num_parameters == ps.num_parameters
        for key, value in self._parameters.items():
            assert key in ps._parameters
            assert value == ps._parameters[key]

        for case in ps._cases:
            if case in self._cases:
                foreign_dict = ps._data[case]
                my_dict = self._data[case]

                for key, val in foreign_dict.items():
                    if key not in my_dict:
                        my_dict[key] = val
                    else:
                        if accumulate:
                            assert isinstance(my_dict[key], list)
                            assert isinstance(foreign_dict[key], list)
                            my_dict[key] = my_dict[key] + val
                        else:
                            raise RuntimeError('Cannot merge dictionaries')
            else:
                self._data[case] = ps._data[case]
                self._cases.append(case)


    def slice(self, f = None, ckey = None, sort = True, **kwargs):

        # f: function that acts as filter on every self._data dict entry matched with parameters
        dofs = self.num_parameters - len(kwargs)
        if not dofs in [1]:
            raise ValueError

        for arg, kval in kwargs.items():
            self._check_parameter(arg, kval)

        matched_cases = list()
        free_params_values = list()
        for case in self._cases:
            num_matches = 0
            for index, pvalue in enumerate(case):
                lkey = self._parameters_ordered[index]
                if lkey in kwargs and kwargs[lkey] == pvalue:
                    num_matches += 1
                else:
                    free_param_type = self._parameters_ordered[index]
                    free_param_value = pvalue
                    assert dofs == 1
            if num_matches == len(kwargs):
                matched_cases.append(case)
                free_params_values.append(free_param_value)

        matched_data = [self._data[case] for case in matched_cases]

        if sort:
            matched_data, free_params_values = zip(*sorted(zip(matched_data, free_params_values), key=lambda pair: pair[1]))

        if ckey is not None:
            if f is not None:
                raise ValueError('Can either provide "ckey" or function, not both')
            matched_data = [data[ckey] for data in matched_data]

            return free_params_values, matched_data

        if f is not None:
            return free_params_values, [f(mdict) for mdict in matched_data]

        return free_params_values, matched_data

    @property
    def num_parameters(self):
        return len(self._parameters)

    def finalize_parameters(self):
        raise NotImplementedError

    def get_global_key(self, **kwargs):
        return self._get_global_key(StringRepresentation = False, **kwargs)

    def _get_global_key(self, StringRepresentation = False, **kwargs):

        for arg, kval in kwargs.items():
            self._check_parameter(arg, kval)

        if not len(kwargs) == len(self._parameters):
            raise KeyError('The specified key ({}) does not contain all registered parameters'.format(kwargs))

        if not StringRepresentation:
            gkey = list()
            for key in self._parameters_ordered:
                gkey.append(kwargs[key])
            gkey = tuple(gkey)
        else:
            gkey = ""
            for i, key in enumerate(self._parameters_ordered):
                gkey += '{}_{}'.format(key, kwargs[key])
                if not i == len(kwargs) - 1:
                    gkey += '_'

        return gkey


    def _check_parameter(self, param, value):

        if not param in self._parameters:
            raise KeyError('Parameter {} has not been registered. We only accept {}'.format(param, tuple(self._parameters.keys())))

        type = self._parameters[param]
        if isinstance(type, list):
            if not value in type:
                raise KeyError
        elif inspect.isclass(type):
            if not isinstance(value, type):
                raise KeyError
        else:
            raise Exception


    def put_dictionary_with_key(self, mdict, gkey, accumulate = False):

        for key, value in mdict.items():
            self.put_with_key(key, value, gkey, accumulate=accumulate)

    def put_dictionary(self, mdict, accumulate = False, **kwargs):

        for key, value in mdict.items():
            self.put(key, value, accumulate=accumulate, **kwargs)

    def put_with_key(self, name, value, gkey, accumulate = False):

        if gkey not in self._data:
            self._data[gkey] = dict()

        if gkey not in self._cases:
            self._cases.append(gkey)

        if accumulate:
            if name in self._data[gkey]:
                self._data[gkey][name].append(value)
            else:
                self._data[gkey][name] = [value]
        else:
            self._data[gkey][name] = [value]


    def put(self, name, value, accumulate=False, **kwargs):

        if not isinstance(name, str):
            raise ValueError

        gkey = self._get_global_key(**kwargs)
        self.put_with_key(name, value, gkey, accumulate=accumulate)

    def get(self, name, **kwargs):

        if not isinstance(name, str):
            raise KeyError

        gkey = self._get_global_key(**kwargs)

        if gkey not in self._data:
            raise KeyError


        if name not in self._data[gkey]:
            raise KeyError

        r = self._data[gkey][name]

        if len(r) == 1:
            r = r[0]

        return r

    def save(self, path):
        file = open(path + '.pickle', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self, path):
        file = open(path + '.pickle', 'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)

    @classmethod
    def FromFile(cls, path):
        database = cls()
        database.load(path)
        return database

    def __repr__(self):

        s = '------------------------------ \n'
        s += 'Registered parameters: \n'
        for item, values in self._parameters.items():
            s += '{} ----- {} \n'.format(item, values)
        s += '------------------------------ \n'
        return s






class ResultsDatabase(object):

    def __init__(self, max_values=2):

        # dict of dicts
        self._dicts = dict()
        self._max_values = max_values
        self._parameters = dict()

    def _get_global_key(self, **kwargs):

        gkey = ''
        for i, (key, value) in enumerate(kwargs.items()):
            gkey += '{}_{}'.format(key, value)
            if not i == len(kwargs) - 1:
                gkey += '_'
        return gkey


    @property
    def num_registered_parameters(self):
        return len(self._parameters)

    def _getdict(self, retrieve=False, **kwargs):

        gkey = self._get_global_key(**kwargs)

        if gkey not in self._dicts and not retrieve:
            self._dicts[gkey] = dict()

            for key, value in kwargs.items():
                if key not in self._parameters:
                    self._parameters[key] = list()
                if value not in self._parameters[key]:
                    self._parameters[key].append(value)



        elif gkey not in self._dicts and retrieve:
            raise KeyError

        return self._dicts[gkey]

    def check_exists(self, **kwargs):

        return self._get_global_key(**kwargs) in self._dicts

    def mark_complete(self, **kwargs):

        d = self._getdict(retrieve=True, **kwargs)
        d['_is_completed_'] = True

    def check_complete(self, **kwargs):

        d = self._getdict(retrieve=True, **kwargs)

        if 'is_completed_' in d and d['_is_completed_']:
            return True
        else:
            return False


    def Storinator(self, **kwargs):

        def f(key, value):
            self.put(key, value, **kwargs)

        return f

    def put(self, key, value, **kwargs):
        d = self._getdict(**kwargs)
        d[key] = value

    def accumulate(self, mkey, f=None, **kwargs):

        for key, value in kwargs.items():
            assert key in self._parameters
            assert value in self._parameters[key]

        results = list()
        for skey in self._dicts:
            counter = 0
            for key_, value_ in kwargs.items():
                lkey = '{}_{}'.format(key_, value_)
                if lkey in skey:
                    counter += 1
            if counter == len(kwargs):
                results.append(self._dicts[skey][mkey])

        if f is not None:
            results = [f(m) for m in results]

        return results

    def get(self, key, **kwargs):
        d = self._getdict(retrieve=True, **kwargs)
        return d[key]

    def save(self, path):
        file = open(path + '.pickle', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()


    def load(self, path):
        file = open(path + '.pickle', 'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)

    @classmethod
    def FromFile(cls, path):
        database = cls()
        return database.load(path)

    def __repr__(self):

        s = '------------------------------ \n'
        s += 'Registered parameters: \n'
        for item, values in self._parameters.items():
            s += '{} ----- {} \n'.format(item, values)
        s += '------------------------------ \n'
        return s
