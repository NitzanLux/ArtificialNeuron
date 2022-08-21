import datetime
import pickle

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
from neuron_simulations.simulation_data_generator_new import SimulationDataGenerator
from train_nets.neuron_network import fully_connected_temporal_seperated
from train_nets.neuron_network import neuronal_model
from utils.general_aid_function import *
from utils.general_variables import *

GOOD_AND_BAD_SIZE = 8

I = 8
BUFFER_SIZE_IN_FILES_VALID = 1


class SimulationData():
    def __init__(self, v, s, index_labels: List, data_label: str):
        assert len(index_labels) == v.shape[0], "labels length does not match to the data length"
        assert v.shape == s.shape, "voltage and spikes are not match"
        self.data_label = data_label
        self.v = v
        self.s = s
        self.index_keys = {k: v for v, k in enumerate(index_labels)}

    def __str__(self):
        return data_label

    def __len__(self):
        assert self.v.size == self.s.size, "spikes array and soma voltage array are inconsistent"
        return self.v.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def is_recording(self):
        return len(self) > 0

    def __getitem__(self, recording_index):
        if recording_index in self.index_keys: recording_index = self.index_keys[recording_index]
        return self.v[recording_index, :], self.s[recording_index, :]

    def get_key(self, index):
        return self.index_keys[index]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            out = pickle.load(f)
        return out


class GroundTruthData(SimulationData):
    def __init__(self, data_files, data_label: str,sort=True):
        # data_files=sorted(data_files)
        self.d_input = []
        data_keys = []
        s, v = [], []
        if sort:
            data_files= sorted(data_files)
        data_generator = SimulationDataGenerator(data_files, buffer_size_in_files=BUFFER_SIZE_IN_FILES_VALID,
                                                 batch_size=1,
                                                 window_size_ms=-1,
                                                 sample_ratio_to_shuffle=1,
                                                 # number_of_files=1,number_of_traces_from_file=2,# todo for debugging
                                                 ).eval()
        for i, data in enumerate(data_generator):
            d_input, d_labels = data
            s_cur, v_cur = d_labels
            s.append(s_cur.cpu().detach().numpy().squeeze())
            v.append(v_cur.cpu().detach().numpy().squeeze())
            self.d_input.append(d_input)
            data_keys.append(data_generator.display_current_fils_and_indexes())
        self.data_files = tuple(data_files)
        s = np.vstack(s)
        v = np.vstack(v)
        self.d_input = np.vstack(self.d_input)
        super().__init__(v, s, index_labels, data_label)

    def __hash__(self):
        return self.data_files.__hash__()

    def __eq__(self, item):
        return self.data_files == item.data_files

    def __getitem__(self, recording_index):
        return self.v[recording_index, :], self.s[recording_index, :], self.d_input[recording_index, ...]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class EvaluationData(SimulationData):
    def __init__(self, ground_truth: GroundTruthData, config):
        self.config = config
        self.ground_truth: ['GroundTruthData'] = ground_truth
        v, s = self.__evaluate_model()
        s = np.vstack(s)
        v = np.vstack(v)
        super().__init__(v, s, ground_truth.data_keys, data_label)
        # self.data_per_recording = [] if recoreded_data is None else recoreded_data

    def __evaluate_model(self):
        assert not self.is_recording(), "evaluation had been done in this object"
        model = self.load_model()
        model.cuda().eval()
        if DATA_TYPE == torch.cuda.FloatTensor:
            model.float()
        elif DATA_TYPE == torch.cuda.DoubleTensor:
            model.double()
        data_keys, s_out, v_out = [], [], []

        for i, data in enumerate(self.ground_truth):
            print(i)
            d_input, s, v = data
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output_s, output_v = model(d_input.cuda().type(DATA_TYPE))
                    # output_s, output_v = model(d_input.cuda().type(torch.cuda.FloatTensor))
                    output_s = torch.nn.Sigmoid()(output_s)
            v_out.append(output_v.cpu().detach().numpy().squeeze())
            s_out.append(output_s.cpu().detach().numpy().squeeze())
        return v_out, s_out

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


class ModelEvaluator():
    def __init__(self, *args: SimulationData):#, use_only_groundtruth=False):
        ground_truth_set = set()
        model_set = set()
        # if not use_only_groundtruth:
        for i in args:
            if isinstance(i, GroundTruthData):
                ground_truth_set.add(i)
            elif isinstance(i, EvaluationData):
                model_set.add(i)
                ground_truth_set.add(i.ground_truth)
            else:
                model_set.add(i)
        self.ground_truths = list(ground_truth_set)
        self.models = list(model_set)
        self.current_good_and_bad_div = None

    def display(self):
        app = dash.Dash()

        auc, fig = self.create_ROC_curve()

        gt_index_dict = {k: i for i, k in enumerate(self.ground_truths)}

        min_shape = min([min([j[1].shape[1] for j in i]) for i in ground_truth])
        slider_arr = [dcc.Slider(
            id='my-slider%d' % i,
            min=0,
            max=len(d) - 1,
            step=1,
            value=len(d) // 2,
            tooltip={"placement": "bottom", "always_visible": True}
        ) for i, d in enumerate(self.ground_truths)]
        dives_arr = [*slider_arr,
                     html.Div(id='slider-output-container', style={'height': '2vh'}),
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
                               'vertical-align': 'middle'}),
                     html.Div([dcc.Markdown('Ground truth\n' + ''.join(
                         ["(%d) %s" % (i, str(k)) for i, k in enumerate(self.ground_truths)]))],
                              style={'border-style': 'solid', 'align': 'center'}),
                     html.Div([dcc.Markdown('Models \n' + ''.join(
                         ["(%d,%d) %s" % (gt_index_dict[k.ground_truth], i, str(k)) for i, k in
                          enumerate(self.ground_truths)]))], style={'border-style': 'solid', 'align': 'center'}),
                     html.Div([
                         dcc.Graph(id='evaluation-graph', figure=go.Figure(),
                                   style={'height': '95vh', 'margin': '0', 'border-style': 'solid',
                                          'align': 'center'})],
                         style={'height': '95vh', 'margin': '0', 'border-style': 'solid', 'align': 'center'}),
                     html.Div([dcc.Markdown('AUC: %0.5f' % auc)]),
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
                Input("mse_window_input", "value"), *[Input('my-slider%d' % 0, 'value')]
            ])
        def update_output(btnm50, btnm10, btnm5, btnm1, btnp1, btnp5, btnp10, btnp50, *values):
            changed_id = [p['prop_id'] for p in callback_context.triggered][0][:-len(".n_clicks")]
            values = [int(v) for v in values]

            if 'btn-m' in changed_id:
                values = [v - int(changed_id[len('btn-m'):]) for v in values]
            elif 'btn-p' in changed_id:
                values = [v + int(changed_id[len('btn-m'):]) for v in values]

            # values = [ max(0, v) for v in values]
            values = [v % len(self.ground_truths[i]) for i, v in enumerate(values)]
            fig = self.display_window(value)
            return *values, 'You have selected "{}"'.format(value), fig

        app.run_server(debug=True, use_reloader=False)

    def create_ROC_curve(self):
        if len(self.data) == 0:
            return 0.5, go.Figure()
        labels_predictions = self.data.get_spikes_for_ROC()
        labels, prediction = labels_predictions[:, 0], labels_predictions[:, 1]
        labels = labels.squeeze().flatten()
        prediction = prediction.squeeze().flatten()
        auc = skm.roc_auc_score(labels, prediction)
        fpr, tpr, _ = skm.roc_curve(labels, prediction)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name='model'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='chance'))
        return auc, fig

    def display_window(self, index):
        v, v_p, s, s_p = self[index]

        fig = make_subplots(rows=3, cols=1,
                            shared_xaxes=True,  # specs = [{}, {},{}],
                            vertical_spacing=0.05, start_cell='top-left',
                            subplot_titles=("voltage", 'running w=%d ' % (general_mse),
                                            "spike probability"), row_heights=[0.6, 0.03, 0.37])
        x_axis = np.arange(v.shape[0])
        # s *= (np.max(s_p) * 1.1)
        # fig.add_trace(go.Scatter(x=x_axis, y=np.convolve(v,np.full((self.config.input_window_size//2,),1./(self.config.input_window_size//2))), name="avg_voltage"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=v, name="voltage"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=v_p, name="predicted voltage"), row=1, col=1)
        fig.add_heatmap(z=running_mse, row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=s, name="spike"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=s_p, name="probability of spike"), row=3, col=1)

        fig.update_layout(  # height=600, width=600,
            title_text="model %s index %d" % (self.config.model_path[-1], index),
            yaxis_range=[-83, -52], yaxis3_range=[0, 1]
            , legend_orientation="h", yaxis2=dict(ticks="", showticklabels=False))
        return fig

    @staticmethod
    def running_mse(v, v_p, mse_window_size):
        mse_window_size = min(v.shape[0], mse_window_size)
        running_mse = np.power(v - v_p, 2)[np.newaxis, :]
        total_mse = np.mean(running_mse)
        if mse_window_size > 2:
            running_mse = uniform_filter1d(running_mse, size=mse_window_size, mode='constant')
            running_mse[:, :(mse_window_size // 2)] = 0
            running_mse[:, -(mse_window_size // 2 - 1):] = 0
        return running_mse, total_mse

    @staticmethod
    def build_and_save(items_path):
        print("start create evaluation", flush=True)
        start_time = datetime.datetime.now()
        if config is None:
            config = configuration_factory.load_config_file(config_path)
        evaluation_engine = ModelEvaluator(config)
        evaluation_engine.evaluate_model(model)
        evaluation_engine.save()
        end_time = datetime.datetime.now()
        print("evaluation took %0.1f minutes" % ((end_time - start_time).total_seconds() / 60.))


#
if __name__ == '__main__':
    path=''
    from utils.general_aid_function import load_files_names
    g = GroundTruthData(load_files_names(path),'test')
    g.save('test.gtest')
    me = ModelEvaluator(g)
    me.display()