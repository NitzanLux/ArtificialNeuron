import datetime
import pickle
import plotly.express as px
import dash
import numpy as np
import plotly.graph_objects as go
import sklearn.metrics as skm
from dash import dcc, callback_context
from dash import html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from scipy.ndimage.filters import uniform_filter1d
import train_nets.neuron_network.recursive_neuronal_model as recursive_neuronal_model
from neuron_simulations.simulation_data_generator_new import parse_sim_experiment_file
from train_nets.neuron_network import fully_connected_temporal_seperated
from train_nets.neuron_network import neuronal_model
from utils.general_aid_function import *
from utils.general_variables import *
from train_nets.configuration_factory import load_config_file
import plotly
import os
import plotly.express as px
from utils.general_variables import *
BATCH_SIZE = 32

cols = px.colors.qualitative.Alphabet

GOOD_AND_BAD_SIZE = 8

I = 8
BUFFER_SIZE_IN_FILES_VALID = 1


class SimulationData():
    def __init__(self, v, s, data_keys: List, data_label: str):
        assert len(data_keys) == v.shape[0], "labels length does not match to the data length"
        assert v.shape == s.shape, "voltage and spikes are not match"
        self.data_label = data_label
        self.v = v
        self.s = s
        self.data_keys = data_keys
        self.keys_dict = {k: v for v, k in enumerate(data_keys)}

    def __str__(self):
        return self.data_label

    def __len__(self):
        assert self.v.size == self.s.size, "spikes array and soma voltage array are inconsistent"
        return self.v.shape[0]

    def __iter__(self):
        for i in sorted(self.keys_dict.keys()):
            yield self[i]

    def is_recording(self):
        return len(self) > 0

    def __getitem__(self, key):
        return self.v[self.keys_dict[key], :], self.s[self.keys_dict[key], :]

    def get_index(self, key):
        return self.keys_dict[key]

    def get_by_index(self, index):
        return self.v[index, :], self.s[index, :]

    def get_key_by_index(self, index):
        return self.data_keys[index]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def __in__(self, key):
        return key in self.keys_dict

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            out = pickle.load(f)
        return out


class GroundTruthData(SimulationData):
    def __init__(self, data_files, data_label: str, sort=True):
        # data_files=sorted(data_files)
        data_keys = []
        self.files_size_dict = {}
        s, v = [], []
        if sort:
            data_files = sorted(data_files)
        for f in data_files:
            X, y_spike, y_soma = parse_sim_experiment_file(f)
            if len(X.shape) == 3:
                X = np.transpose(X, axes=[2, 0, 1])
            else:
                X = X[np.newaxis, ...]
            self.files_size_dict[f] = X.shape[0]
            for i in range(X.shape[0]):
                s.append(y_spike.T[i, ...])
                v.append(y_soma.T[i, ...])
                data_keys.append((f, i))
        self.data_files = tuple(data_files)
        self.files_short_names = {i: i for i in self.data_files}
        self.crete_dense_representation_of_files()
        self.files_short_names = {v: k for k, v in self.files_short_names.items()}
        s = np.vstack(s)
        v = np.vstack(v)
        super().__init__(v, s, data_keys, data_label)

    def translate_tuple_to_files(self, f, i):
        f = get_file_from_shortmane(f)
        return (f, i) if (f, i) in self else None

    def get_file_shortname(self, f):
        return list(self.files_short_names.keys())[list(self.files_short_names.values()).index(f)]

    def get_file_from_shortmane(self, f):
        return self.files_short_names[f]

    def crete_dense_representation_of_files(self):
        res = self.findstem()
        while len(res) > 0:
            for f in self.data_files:
                self.files_short_names[f] = self.files_short_names[f].replace(res, '')
            res = self.findstem()

    def findstem(self):
        # Determine size of the array
        files_short_names = list(self.files_short_names.values())
        n = len(files_short_names)

        # Take first word from array
        # as reference
        s = files_short_names[0]
        l = len(s)
        res = ""
        for i in range(l):
            for j in range(i + 1, l + 1):
                # generating all possible substrings
                # of our reference string files_short_names[0] i.e s
                stem = s[i:j]
                k = 1
                for k in range(1, n):
                    # Check if the generated stem is
                    # common to all words
                    if stem not in files_short_names[k]:
                        break
                # If current substring is present in
                # all strings and its length is greater
                # than current result
                if (k + 1 == n and len(res) < len(stem)):
                    res = stem
        return res

    def __hash__(self):
        return self.data_files.__hash__()

    def __eq__(self, item: 'GroundTruthData'):
        return self.data_files == item.data_files

    def get_evaluation_input_per_file(self, f, batch_size=8):
        assert f in self.data_files, "file not exists in this simulation."
        X, _, __ = parse_sim_experiment_file(f)
        X = torch.from_numpy(X)
        X = np.transpose(X, axes=[2, 0, 1])
        for i in range(0, self.files_size_dict[f], batch_size):
            l_range = i
            h_range = min(l_range + batch_size, self.files_size_dict[f])
            yield (X[l_range:h_range, ...], [(f, i) for i in range(l_range, h_range)]), self.files_size_dict[
                f] > batch_size  # if the buffer size is bigger then the file size then we need to accumulate.

    def get_evaluation_input(self, batch_size=8):
        buffer = []
        for f in self.data_files:
            for i, cond in self.get_evaluation_input_per_file(f, batch_size):
                if cond:
                    yield i
                else:
                    buffer[0] = np.vstack((buffer[0], i[0]))
                    buffer[1] = buffer[1] + i[1]
                    if len(buffer[1]) == batch_size:
                        yield buffer
                        buffer = []


class EvaluationData(SimulationData):
    def __init__(self, ground_truth: GroundTruthData, config):
        self.config = config
        self.ground_truth: ['GroundTruthData'] = ground_truth
        v, s, data_keys = self.__evaluate_model()
        s = np.vstack(s)
        v = np.vstack(v)
        assert sum([i != j for i, j in zip(data_keys,
                                           self.ground_truth.data_keys)]) == 0, "Two data keys of ground_truth and model evaluation are different." + "\n" + "\n".join(
            [(str(i[1]) + "\t|\t" + str(j[1])) if i != j else '\b' for i, j in
             zip(data_keys, self.ground_truth.data_keys)])
        super().__init__(v, s, data_keys, config.model_tag)
        # self.data_per_recording = [] if recoreded_data is None else recoreded_data

    def __evaluate_model(self):
        # assert not self.is_recording(), "evaluation had been done in this object"
        model = self.load_model()
        model.cuda().eval() if USE_CUDA else model.cpu().eval()
        if DATA_TYPE == torch.cuda.FloatTensor or DATA_TYPE == torch.FloatTensor:
            model.float()
        elif DATA_TYPE == torch.cuda.DoubleTensor or DATA_TYPE==torch.DoubleTensor:
            model.double()
        data_keys, s_out, v_out = [], [], []
        i = 0
        for inputs, keys in self.ground_truth.get_evaluation_input(batch_size=BATCH_SIZE):
            i += 1
            with torch.no_grad():
                inputs.cuda() if USE_CUDA else inputs.cpu()
                output_s, output_v = model(inputs.type(DATA_TYPE))
                output_s = torch.nn.Sigmoid()(output_s)
            v_out.append(output_v.cpu().detach().numpy().squeeze())
            s_out.append(output_s.cpu().detach().numpy().squeeze())
            data_keys = data_keys + keys
            # print([i[1] for i in data_keys],flush=True)
        v_out = np.vstack(v_out)
        s_out = np.vstack(s_out)
        return v_out, s_out, data_keys

    def load_model(self):
        print("loading model...", flush=True)
        if self.config is None: return None
        if self.config.architecture_type == "DavidsNeuronNetwork":
            model = davids_network.DavidsNeuronNetwork.load(self.config)
        elif self.config.architecture_type == "FullNeuronNetwork":
            model = fully_connected_temporal_seperated.FullNeuronNetwork.load(self.config)
        elif "network_architecture_structure" in self.config and self.config.network_architecture_structure == "recursive":
            model = recursive_neuronal_model.RecursiveNeuronModel.load(self.config)
        else:
            model = neuronal_model.NeuronConvNet.build_model_from_config(self.config)
        print("model parmeters: %d" % model.count_parameters())
        return model

    def running_mse(self, index, mse_window_size):
        if not hasattr(self, '__running_mse'):
            v, v_p = self.ground_truth[index], self[index]
            mse_window_size = min(v.shape[0], mse_window_size)
            self.__running_mse = np.power(v - v_p, 2)[np.newaxis, :]
            self.__total_mse = np.mean(self.__running_mse)

        if mse_window_size == 1:
            running_mse = self.__running_mse
        else:
            running_mse = uniform_filter1d(self.__running_mse, size=mse_window_size, mode='constant')
            running_mse[:, :(mse_window_size // 2)] = 0
            running_mse[:, -(mse_window_size // 2 - 1):] = 0
        return running_mse, self.__total_mse

    def get_ROC_data(self):
        # labels_predictions = self.data.get_spikes_for_ROC()
        # labels, prediction = labels_predictions[:, 0], labels_predictions[:, 1]
        target = self.ground_truth.s
        prediction = self.s
        target = target[:, target.shape[1] - prediction.shape[1]:]
        target = target.squeeze().flatten()
        prediction = prediction.squeeze().flatten()
        auc = skm.roc_auc_score(target, prediction)
        fpr, tpr, _ = skm.roc_curve(target, prediction)
        return auc, fpr, tpr


class ModelEvaluator():
    def __init__(self, *args: SimulationData):  # , use_only_groundtruth=False):
        ground_truth_set = set()
        model_set = set()
        # if not use_only_groundtruth:
        for i in args:
            if isinstance(i, GroundTruthData):
                ground_truth_set.add(i)
                print(i.v.shape)
            elif isinstance(i, EvaluationData):
                model_set.add(i)
                ground_truth_set.add(i.ground_truth)
            else:
                model_set.add(i)
        # f_name_set=None
        # for i,gt in enumerate(self.ground_truths):
        #     if f_name_set is None:
        #         f_name_set=set(gt.file)
        self.file_names_short_names = set()
        self.ground_truths = list(ground_truth_set)
        self.models = list(model_set)
        self.current_good_and_bad_div = None
        self.__currnt_value = None
        self.gt_index_dict = {k: i for i, k in enumerate(self.ground_truths)}
        self.cur_restyleData= {}
    def locking_function(self, value, locking):
        if self.__currnt_value is None:
            self.__currnt_value = value
            return self.__currnt_value
        if locking == 'free':
            self.__currnt_value = value
            return self.__currnt_value
        elif locking == 'locked':
            step = sum([j - i for i, j in zip(self.__currnt_value, value)])
            new_value = [step + v for v in self.__currnt_value]
            self.__currnt_value = new_value
            return self.__currnt_value
        else:
            changed_id = sum([(i[1] - i[0]) * k for k, i in enumerate(zip(self.__currnt_value, value))])
            f_i = self.ground_truths[changed_id].get_file_shortname(
                self.ground_truths[changed_id].get_key_by_index(value[changed_id]))
            f_is = [gt.get_file_from_shortmane(f_i) for gt in self.ground_truths]
            for i in range(len(f_is)):
                if f_is[i] is not None:
                    values[i] = f_is[i]
            self.__currnt_value = values
            return self.__currnt_value

    def display(self):
        app = dash.Dash()

        fig, AUC_arr = self.create_ROC_curve()

        min_shape = min([i.v.shape[1] for i in self.ground_truths])
        slider_arr = [html.Div([dcc.Markdown('(%d)' % (i),
                                             style={"verticalAlign": "top", 'width': '2vw', 'display': 'inline-block',
                                                    'text-align': 'top'})
                                   , html.Div(
                [dcc.Slider(id='my-slider%d' % i, min=0, max=len(d) - 1, step=1, value=len(d) // 2,
                            tooltip={"placement": "bottom", "always_visible": True})],
                style={'align': 'center', 'width': '96vw', 'display': 'inline-block', "margin-left": "1px"})
                                ], style={'height': '5vh', 'width': '100vw', "verticalAlign": "top"}) for i, d in
                      enumerate(self.ground_truths)]
        dives_arr = [
            html.Div(id='slider-output-container', style={"white-space": "pre"}), *slider_arr,
            dcc.RadioItems(id='choose_locking', options=[
                {'label': 'free', 'value': 'free'},
                {'label': 'locked', 'value': 'locked'},
                {'label': 'file locked', 'value': 'f_locked'}], persistence=True, value='free',
                           style={'display': 'inline-block', 'align': 'center'}),
            html.Div([

                html.Button('-50', id='btn-m50', n_clicks=0,
                            style={'margin': '1', 'align': 'center', 'vertical-align': 'middle',
                                   "margin-left": "10px"}),
                html.Button('-10', id='btn-m10', n_clicks=0,
                            style={'margin': '1', 'align': 'center', 'vertical-align': 'middle',
                                   "margin-left": "10px"}),
                html.Button('-5', id='btn-m5', n_clicks=0,
                            style={'margin': '1', 'align': 'center', 'vertical-align': 'middle',
                                   "margin-left": "10px"}),
                html.Button('-1', id='btn-m1', n_clicks=0,
                            style={'margin': '1', 'align': 'center', 'vertical-align': 'middle',
                                   "margin-left": "10px"}),
                html.Button('+1', id='btn-p1', n_clicks=0,
                            style={'margin': '1', 'align': 'center', 'vertical-align': 'middle',
                                   "margin-left": "10px"}),
                html.Button('+5', id='btn-p5', n_clicks=0,
                            style={'margin': '1', 'align': 'center', 'vertical-align': 'middle',
                                   "margin-left": "10px"}),
                html.Button('+10', id='btn-p10', n_clicks=0,
                            style={'margin': '1', 'align': 'center', 'vertical-align': 'middle',
                                   "margin-left": "10px"}),
                html.Button('+50', id='btn-p50', n_clicks=0,
                            style={'margin': '1', 'align': 'center', 'vertical-align': 'middle',
                                   "margin-left": "10px"}),
                ' mse window size:',
                dcc.Input(
                    id="mse_window_input",
                    type="number",
                    value=100,
                    step=1,
                    min=1,
                    debounce=True,
                    max=min_shape,
                    placeholder="Running mse window size",
                    style={'margin': '10', 'align': 'center', 'vertical-align': 'middle',
                           "margin-left": "10px"}
                )
            ], style={'width': '100vw', 'margin': '1', 'border-style': 'solid', 'align': 'center',
                      "verticalAlign": "top"}),
            html.Div([dcc.Markdown(('*Ground truth*\t ' + '\t\t\t\n'.join(
                ["\t(%d) %s" % (i, str(k)) for i, k in enumerate(self.ground_truths)])),
                                   style={"white-space": "pre", 'width': '49vw', 'display': 'inline-block',
                                          'margin': '0.1', 'border-style': 'solid', "verticalAlign": "top"})
                         , dcc.Markdown('*Models*\t ' + '\t\t\t\n'.join(
                    ["\t(%d,%d) %s" % (self.gt_index_dict[k.ground_truth], i, str(k)) for i, k in
                     enumerate(self.models)]), style={"white-space": "pre", 'width': '49vw', 'margin': '0.1',
                                                      'border-style': 'solid', 'display': 'inline-block',
                                                      "verticalAlign": "top"})],
                     style={'width': '100vw', 'margin': '1', 'border-style': 'solid', "verticalAlign": "top",
                        "white-space": "pre"
        }),
            html.Div([
                dcc.Graph(id='evaluation-graph', figure=go.Figure(),
                          style={'height': '95vh', 'margin': '0', 'border-style': 'solid',
                                 'align': 'center'})],
                style={'height': '95vh', 'margin': '0', 'border-style': 'solid', 'align': 'center'})
            , html.Div([dcc.Markdown("\n".join(AUC_arr))]),
            html.Div([dcc.Graph(id='eval-roc', figure=fig,
                                style={'height': '95vh', 'margin': '0', 'border-style': 'solid',
                                       'align': 'center'})],
                     ),
        ]
        app.layout = html.Div(dives_arr)

        @app.callback(
            *[Output('my-slider%d' % i, 'value') for i in range(len(self.ground_truths))],
            Output('slider-output-container', "children"),
            Output('evaluation-graph', 'figure'),
            [
                Input('btn-m50', 'n_clicks'),
                Input('btn-m10', 'n_clicks'),
                Input('btn-m5', 'n_clicks'),
                Input('btn-m1', 'n_clicks'),
                Input('btn-p1', 'n_clicks'),
                Input('btn-p5', 'n_clicks'),
                Input('btn-p10', 'n_clicks'),
                Input('btn-p50', 'n_clicks'),
                Input("mse_window_input", "value"),
                Input("choose_locking", "value"),
                *[Input('my-slider%d' % i, 'value') for i in range(len(self.ground_truths))]
            ])
        def update_output(btnm50, btnm10, btnm5, btnm1, btnp1, btnp5, btnp10, btnp50, mse_window_size,locking, *values):
            changed_id = [p['prop_id'] for p in callback_context.triggered][0][:-len(".n_clicks")]
            values = [int(v) for v in values]
            print(locking)
            if 'btn-m' in changed_id:
                values = [v - int(changed_id[len('btn-m'):]) for v in values]
            elif 'btn-p' in changed_id:
                values = [v + int(changed_id[len('btn-m'):]) for v in values]
            else:
                values = self.locking_function(values, locking)
            # values = [v % len(self.ground_truths[i]) for i, v in enumerate(values)]
            values = [min(v, len(self.ground_truths[i]) - 1) for i, v in enumerate(values)]
            values = [max(v, 0) for i, v in enumerate(values)]
            fig = self.display_window(values, mse_window_size)
            # if restyleData is not None:
            #     print(restyleData)
            #     for i,v in enumerate(restyleData[1]):
            #         restyleData[0]['visible'][i]='legendonly'
            #         self.cur_restyleData[v]=False #restyleData[0]['visible'][i]
            # restyle_data_out = [{'visible':sorted(list(self.cur_restyleData.values()))},sorted(list(self.cur_restyleData.keys()))] if len(list(self.cur_restyleData.keys())) else None
            # print(restyle_data_out)

            return *values, 'You have selected \n' + '\n'.join(
                [str((os.path.basename(gt.get_key_by_index(values[i])[0]), gt.get_key_by_index(values[i])[1])) for i, gt
                 in enumerate(self.ground_truths)]), fig

        app.run_server(debug=True, use_reloader=False)

    def create_ROC_curve(self):
        fig = go.Figure()
        AUC_arr = []
        for i, m in enumerate(self.models):
            auc, fpr, tpr = m.get_ROC_data()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name="(gt: %s model: %s)" % (self.gt_index_dict[m.ground_truth], self.models[i].data_label)))
            AUC_arr.append((auc,"(gt: %s model: %s) AUC: %.4f" % (self.gt_index_dict[m.ground_truth], self.models[i].data_label, auc)))
        AUC_arr= sorted(AUC_arr,key=lambda x: x[0])
        AUC_arr=[v[1]+("\n" if i%1==0 else '') for i,v in enumerate(AUC_arr)]
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='chance'))
        fig.update_layout(title_text="ROC", showlegend=True)
        return fig, AUC_arr

    def display_window(self, indexes, mse_window_size):
        fig = make_subplots(rows=3, cols=1,
                            shared_xaxes=True,  # specs = [{}, {},{}],
                            vertical_spacing=0.05, start_cell='top-left',
                            subplot_titles=("voltage",
                                            "spike probability"), row_heights=[0.6, 0.37, 0.03])
        x_axis_gt = None
        for j, gt in enumerate(self.ground_truths):
            v, s = gt.get_by_index(indexes[j])
            x_axis_gt = np.arange(v.shape[0])
            fig.add_trace(
                go.Scatter(x=x_axis_gt, y=v, legendgroup='gt%d' % j, name="%s" % (self.ground_truths[j]),
                           line=dict(width=2, color=cols[j])),
                row=1, col=1)

            fig.add_trace(go.Scatter(x=x_axis_gt, y=s, legendgroup='gt%d' % j, showlegend=False, name="(%d)" % (j),
                                     line=dict(width=2, color=cols[j])), row=2, col=1)
            # fig['data'][j*2+1]['line']['color']=fig['data'][j*2]['line']['color']
        mse_matrix = []
        y = []
        for j, m in enumerate(self.models):
            v, s = m.get_by_index(indexes[self.gt_index_dict[m.ground_truth]])
            x_axis = np.arange(x_axis_gt.shape[0] - v.shape[0], x_axis_gt.shape[0])
            y.append("(gt: %s model: %d)" % (self.gt_index_dict[m.ground_truth], j))
            fig.add_trace(go.Scatter(x=x_axis, y=v, legendgroup='model%d' % (j + len(self.ground_truths)),
                                     name="(gt: %s model: %s)" % (self.gt_index_dict[m.ground_truth], self.models[j].data_label),
                                     line=dict(width=2, color=cols[(j + len(self.ground_truths))])), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=x_axis, y=s, showlegend=False, legendgroup='model%d' % (j + len(self.ground_truths)),
                           name="(gt: %s model: %s)" % (self.gt_index_dict[m.ground_truth], self.models[j].data_label),
                           line=dict(width=2, color=cols[(j + len(self.ground_truths))])), row=2, col=1)
            # mse_matrix.append(m.running_mse(indexes[self.gt_index_dict[m.ground_truth]], mse_window_size))
        if len(mse_matrix) > 0:
            mse_matrix = np.vstack(mse_matrix)
            fig.add_heatmap(z=mse_matrix, row=3, col=1, y=y)
        fig.update_layout(  # height=600, width=600,
            # title_text="model %s index %d" % (self.config.model_path[-1], index),
            yaxis_range=[-83, -52], yaxis2_range=[0, 1]  # , legend_orientation="h"
            , yaxis2=dict(ticks="", showticklabels=False))
        return fig


def create_gt_and_save(folder, name):
    files = get_files_by_filer_from_dir(folder)
    print('number of files:', len(files), flush=True)
    g = GroundTruthData(files, name)
    g.save(os.path.join("evaluations", 'ground_truth', name + ".gteval"))
    return g


def create_model_evaluation(gt_name, model_name):
    gt_path = os.path.join("evaluations", 'ground_truth', gt_name + ".gteval")
    dest_path = os.path.join("evaluations", 'models', gt_name)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    config = load_config_file(os.path.join(MODELS_DIR, model_name, model_name + ".config"))
    print('load config for %s' % model_name)
    g = EvaluationData(GroundTruthData.load(gt_path), config)

    g.save(os.path.join(dest_path, model_name + ".meval"))


def run_test():
    #
    # if __name__ == '__main__':
    # from utils.general_aid_function import load_files_names
    # path = r"C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\data\L5PC_NMDA_validation"
    # g = GroundTruthData(get_files_by_filer_from_dir(path), 'test1')
    # g.save('test.gtest')
    #
    # path = r"C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\data\L5PC_NMDA_test"
    # g3 = GroundTruthData(get_files_by_filer_from_dir(path), 'test2')
    # g3.save('test1.gtest')
    # path = r"C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\data\valid"
    # g4 = GroundTruthData(get_files_by_filer_from_dir(path), 'test3')
    # g4.save('test2.gtest')

    from utils.general_aid_function import load_files_names
    g=[]
    b_p = r"C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project"
    g0 = GroundTruthData.load(r"evaluations\ground_truth\davids_ergodic_validation.gteval")
    g1 = GroundTruthData.load(r"evaluations\ground_truth\reduction_ergodic_validation.gteval")
    for p in [r"evaluations\models\davids_ergodic_validation",r"evaluations\models\reduction_ergodic_validation"]:
        for i in os.listdir(p):
           g.append(EvaluationData.load(os.path.join(p,i)))
    # g2 = EvaluationData.load(
    #     "evaluations/models/davids_ergodic_test/davids_2_NAdam___2022-08-15__15_02__ID_64341.meval")
    me = ModelEvaluator(g0, g1,*g)
    me.display()
