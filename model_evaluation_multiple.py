import configuration_factory
from neuron_network import neuronal_model
import pickle
from typing import Iterable
import sklearn.metrics as skm
import dash
import numpy as np
import plotly.graph_objects as go
import torch
from dash import dcc, callback_context
from dash import html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from tqdm import tqdm
from simulation_data_generator import SimulationDataGenerator
import neuron_network.node_network.recursive_neuronal_model as recursive_neuronal_model
from general_aid_function import *
from neuron_network import neuronal_model
import plotly.express as px
import datetime
import json
from general_variables import *
from get_neuron_modle import get_L5PC
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import find_peaks
import dash_bootstrap_components as dbc
from neuron_network import fully_connected_temporal_seperated

GOOD_AND_BAD_SIZE = 8

I = 8
BUFFER_SIZE_IN_FILES_VALID = 1


class EvaluationData():
    def __init__(self,ground_truth=None, recoreded_data=None,data_labels=None):
        self.ground_truth= [] if ground_truth is None else ground_truth
        self.data_labels= [] if data_labels is None else data_labels
        self.data_per_recording = [] if recoreded_data is None else recoreded_data

    def clear(self):
        self.data_per_recording = []
        self.data_labels=[]
        self.data_per_recording=[]


    def is_recording(self):
        return len(self) > 0

    def __getitem__(self, recording_index):
        return self.__get_item_by_recording_index(recording_index)
        # assert isinstance(is_spike, bool) or isinstance(is_spike, Iterable) or (
        #         isinstance(is_spike, int) and 0 <= is_spike <= 1) or (
        #                isinstance(is_spike, str) in ('s', 'v')), "cannot assign value to the second dimension"
        # assert isinstance(is_predicted, bool) or isinstance(is_predicted, Iterable) or (
        #         isinstance(is_predicted, int) and 0 <= is_spike <= 1), "cannot assign value to thired dimension"
        # is_predicted, is_spike = self.__cast_indexing(is_predicted, is_spike)
        # return self.data_per_recording[recording_index][is_spike][is_predicted]

    def __get_item_by_recording_index(self, recording_index):
        output=[(self.ground_truth[recording_index][0],self.ground_truth[recording_index][1],None)]
        for data in self.data_labels:
            offset=self.ground_truth#todo work
        # return self.data_per_recording[recording_index][0][0], self.data_per_recording[recording_index][0][1], \
        #        self.data_per_recording[recording_index][1][0], self.data_per_recording[recording_index][1][1]

    def extend(self, v_arr, v_pred_arr, s_arr, s_pred_arr):
        for i in zip(v_arr, v_pred_arr, s_arr, s_pred_arr):
            self.append(*i)

    def append(self, v, v_pred, s, s_pred):
        input_data = []
        if len(v.shape) > 1:
            for i in range(v.shape[0]):
                input_data.append([[v[i, :], v_pred[i, :]], [s[i, :], s_pred[i, :]]])
            self.data_per_recording.extend(input_data)
        else:
            input_data = [[v, v_pred], [s, s_pred]]
            self.data_per_recording.append(input_data)

    def __setitem__(self, recording_index, is_spike, is_predicted, value):
        is_predicted, is_spike = self.__cast_indexing(is_predicted, is_spike)

        self.data_per_recording[recording_index][is_spike][is_predicted] = value
        return value

    def flatten_batch_dimensions(self):
        new_data = EvaluationData()
        for i, d in enumerate(self):
            v, vp, s, sp = d
            if len(v.shape) > 1:
                if v.shape[0] > 1:
                    for j in range(v.shape[0]):
                        new_data.append(v[j, :], vp[j, :], s[j, :], sp[j, :])
                else:
                    new_data.append(v[0, :], vp[0, :], s[0, :], sp[0, :])
            else:
                new_data.append(v, vp, s, sp)
        self.data_per_recording = new_data.data_per_recording

    def __len__(self):
        return len(self.data_per_recording)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_spikes_for_ROC(self):
        output = []
        for i in self:
            _, __, s_arr, s_p_arr = i
            for s, s_p in zip(s_arr, s_p_arr):
                output.append((s, s_p))
        return np.array(output)

    @staticmethod
    def __cast_indexing(is_predicted, is_spike):
        if isinstance(is_spike, Iterable):
            is_spike_new = []
            for i in is_spike:
                if isinstance(i, str):
                    i = 's' == i
                is_spike_new.append(int(i))
            is_spike = is_spike_new
        else:
            if isinstance(is_spike, str):
                is_spike = 's' == is_spike
            is_spike = int(is_spike)
        if isinstance(is_predicted, Iterable):
            is_predicted_new = []
            for i in is_predicted:
                is_predicted_new.append(int(i))
            is_predicted = is_predicted_new
        else:
            is_predicted = int(is_predicted)
        return is_predicted, is_spike


class ModelEvaluator():
    def __init__(self, config, is_validation=True):
        self.data = EvaluationData()
        self.config = config
        self.is_validation = is_validation
        self.current_good_and_bad_div=None


    def evaluate_model(self, model=None):
        assert not self.data.is_recording(), "evaluation had been done in this object"
        if model is None:
            model = self.load_model()
        if model is not None:
            model.cuda().eval()
            if DATA_TYPE == torch.cuda.FloatTensor:
                model.float()
            elif DATA_TYPE == torch.cuda.DoubleTensor:
                model.double()

        data_generator = self.load_data_generator(self.config, self.is_validation)
        for i, data in enumerate(data_generator):
            print(i)
            d_input, d_labels = data
            s, v = d_labels
            if model is not None:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        output_s, output_v = model(d_input.cuda().type(DATA_TYPE))
                        # output_s, output_v = model(d_input.cuda().type(torch.cuda.FloatTensor))
                        output_s = torch.nn.Sigmoid()(output_s)
                output_v=output_v.cpu().detach().numpy().squeeze()
                output_s=output_s.cpu().detach().numpy().squeeze()
            else:
                output_v=zeros_like(v.cpu.detach().numpy().squeeze())
                output_s=zeros_like(v.cpu.detach().numpy().squeeze())
            self.data.append(v.cpu().detach().numpy().squeeze(), output_v,
                             s.cpu().detach().numpy().squeeze(), output_s)

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
            model = neuronal_model.NeuronConvNet.build_model_from_self.config(self.config)
        print("model parmeters: %d" % model.count_parameters())
        return model
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

    def display(self):
        app = dash.Dash()
        auc, fig = self.create_ROC_curve()
        app.layout = html.Div([
            dcc.Slider(
                id='my-slider',
                min=0,
                max=len(self.data) - 1,
                step=1,
                value=len(self.data) // 2,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
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
                    max=self[0][0].shape[0],
                    placeholder="Running mse window size",style={'margin': '10', 'align': 'center', 'vertical-align': 'middle',
                                   "margin-left": "10px"}
                )
            ], style={'width': '100vw', 'margin': '1', 'border-style': 'solid', 'align': 'center',
                      'vertical-align': 'middle'}),
            html.Div([
                dcc.Graph(id='evaluation-graph', figure=go.Figure(),
                          style={'height': '95vh', 'margin': '0', 'border-style': 'solid', 'align': 'center'})],
                style={'height': '95vh', 'margin': '0', 'border-style': 'solid', 'align': 'center'}),
            html.Div([dcc.Markdown('AUC: %0.5f' % auc)]),
            html.Div([dcc.Graph(id='eval-roc', figure=fig,
                                style={'height': '95vh', 'margin': '0', 'border-style': 'solid', 'align': 'center'})],
                     ),
            html.Div([' mse window size:',dcc.Input(
                    id="mse_window_input_good_and_bad",
                    type="number",
                    value=100,
                    step=1,
                    max=self[0][0].shape[0],
                    min=1,
                    debounce=True,
                    placeholder="Running mse window size".format("number"),style={'margin': '10', 'align': 'center', 'vertical-align': 'middle',
                                   "margin-left": "10px"}
                ),' margin size:',dcc.Input(
                    id="margin_good_and_bad",
                    type="number",
                    value=100,
                    step=1,
                    max=1000,
                    min=1,
                    debounce=True,
                    placeholder="Running mse window size".format("number"),style={'margin': '10', 'align': 'center', 'vertical-align': 'middle',
                                   "margin-left": "10px"}
                ),html.Div(id="good_and_bad",children=[self.display_good_and_bad()])],
                     ),

        ])
        @app.callback(
            Output('good_and_bad','children'),
            Input("mse_window_input_good_and_bad", "value"),
            Input("margin_good_and_bad", "value")
        )
        def update_good_and_bad(window_size,margin):
            return self.display_good_and_bad(mse_window_size=window_size,margin=margin)
        @app.callback(
            Output('my-slider', 'value'),
            Output('slider-output-container', "children"),
            Output('evaluation-graph', 'figure'),
            [Input('my-slider', 'value'),
             Input('btn-m50', 'n_clicks'),
             Input('btn-m10', 'n_clicks'),
             Input('btn-m5', 'n_clicks'),
             Input('btn-m1', 'n_clicks'),
             Input('btn-p1', 'n_clicks'),
             Input('btn-p5', 'n_clicks'),
             Input('btn-p10', 'n_clicks'),
             Input('btn-p50', 'n_clicks'),
             Input("mse_window_input", "value")
             ])
        def update_output(value, btnm50, btnm10, btnm5, btnm1, btnp1, btnp5, btnp10, btnp50,mse_window_size):
            changed_id = [p['prop_id'] for p in callback_context.triggered][0][:-len(".n_clicks")]
            value = int(value)

            if 'btn-m' in changed_id:
                value -= int(changed_id[len('btn-m'):])
            elif 'btn-p' in changed_id:
                value += int(changed_id[len('btn-m'):])
            value = max(0, value)
            value = min(value, len(self.data))
            fig = self.display_window(value,mse_window_size)
            return value, 'You have selected "{}"'.format(value), fig

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


    def display_window(self, index, mse_window_size=100):
        v, v_p, s, s_p = self[index]
        running_mse,general_mse = self.running_mse(v,v_p,mse_window_size)

        fig = make_subplots(rows=3, cols=1,
                            shared_xaxes=True, #specs = [{}, {},{}],
                            vertical_spacing=0.05, start_cell='top-left',
                            subplot_titles=("voltage",'running mse w=%d mse=%0.2f'%(mse_window_size,general_mse), "spike probability"), row_heights=[0.6, 0.03, 0.37])
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
            ,legend_orientation="h",yaxis2=dict(ticks="",showticklabels=False))
        return fig

    def display_good_and_bad(self, number_of_good_and_bad_examples=GOOD_AND_BAD_SIZE, mse_window_size=100,margin=20):
        if mse_window_size is None:
            return self.current_good_and_bad_div
        good_examples,bad_examples=self.find_good_and_bad_examples(number_of_good_and_bad_examples,mse_window_size)
        ncols= 3 if number_of_good_and_bad_examples%3==0 else 4
        nrows=number_of_good_and_bad_examples//ncols+(number_of_good_and_bad_examples%ncols>0)
        nrows*=2
        row_stayle={ 'width': '24vw','margin': '0', 'align': 'center', 'display': 'inline-block', "margin-left": "0px"}
        dives=[]
        h_box=[]
        v_box=[]
        current_row=0
        h_box.append(html.Div(children='bad examples'))

        for i in range(bad_examples.shape[0]):
            if i // ncols != current_row:
                h_box.append(html.Div(v_box, style={'width': '100vw', 'margin': '1', 'align': 'center',
                                                    'vertical-align': 'middle'}))
                v_box = []
                current_row += 1
            v_box.append(dcc.Graph(
                                   figure=self.generate_figure_good_or_bad(bad_examples[i, 0], bad_examples[i, 2],
                                                                           bad_examples[i, 1], mse_window_size,margin=margin),
                                   style=row_stayle))
        if len(v_box)>0:
            h_box.append(html.Div(v_box, style={'width': '100vw', 'margin': '1', 'align': 'center',
                                                'vertical-align': 'middle'}))
        dives.append(html.Div(h_box, style={'width': '100vw', 'margin': '1', 'align': 'center','border-style': 'solid'}))

        h_box = []
        v_box = []
        current_row = 0
        h_box.append(html.Div(children='good examples'))
        for i in range(good_examples.shape[0]):
            if i // ncols != current_row:
                h_box.append(dbc.Row(v_box))
                v_box = []
                current_row += 1
            v_box.append(dcc.Graph(
                figure=self.generate_figure_good_or_bad(good_examples[i, 0], good_examples[i, 2], good_examples[i, 1],
                                                        mse_window_size,margin=margin),
                style=row_stayle))
        if len(v_box) > 0:
            h_box.append(html.Div(v_box, style={'width': '100vw', 'margin': '1', 'align': 'center',
                                                'vertical-align': 'middle', "margin-left": "10px"}))
        dives.append(
            html.Div(h_box, style={'width': '100vw', 'margin': '1', 'align': 'center', 'border-style': 'solid'}))
        self.current_good_and_bad_div =  html.Div(dives)
        return self.current_good_and_bad_div


    def find_good_and_bad_examples(self,number_of_good_and_bad_examples,mse_window_size=100):
        good_examples=[]
        bad_examples=[]
        for i in range(len(self)):
            v, v_p, s, s_p = self[i]
            running_mse,general_mse = self.running_mse(v, v_p, mse_window_size)
            running_mse=running_mse[:,(mse_window_size // 2):-(mse_window_size // 2 - 1)][0,:]
            peaks, _ = find_peaks((-running_mse),distance=(mse_window_size // 2)+(mse_window_size % 2))
            for peak in peaks:
                bad_examples.append((running_mse[peak],i,peak))
            peaks, _ = find_peaks(running_mse, distance=(mse_window_size // 2) + (mse_window_size % 2))
            for peak in peaks:
                good_examples.append((running_mse[peak],i,peak))
        good_examples = np.array(good_examples)
        g_ind = np.argpartition(-good_examples[:,0], -number_of_good_and_bad_examples)[-number_of_good_and_bad_examples:]
        bad_examples=np.array(bad_examples)
        b_ind = np.argpartition(bad_examples[:,0], -number_of_good_and_bad_examples)[-number_of_good_and_bad_examples:]
        return good_examples[g_ind,:],bad_examples[b_ind,:]

    def generate_figure_good_or_bad(self, mse, first_time_point, sim_index, window_size,margin=0):
        v, v_p, s, s_p = self[int(sim_index)]
        first_time_point=int(first_time_point)

        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,  # specs = [{}, {},{}],
                            vertical_spacing=0.05, start_cell='top-left',
                            subplot_titles=("voltage mse %0.3f" % mse, "spike probability"),
                            row_heights=[0.6, 0.4])
        x_axis = np.arange(v.shape[0])
        fig.add_trace(go.Scatter(x=x_axis, y=v, name="voltage"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=v_p, name="predicted voltage"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=s, name="spike"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=s_p, name="probability of spike"), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[first_time_point, first_time_point, first_time_point+window_size, first_time_point+window_size,first_time_point],
            y=[-90, -45, -45, -90,-90], fill="toself",opacity=0.3,fillcolor='orange',hoverinfo='skip',line=dict(width=0),mode='lines',showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(
            x=[first_time_point, first_time_point, first_time_point+window_size, first_time_point+window_size,first_time_point],
            y=[0, 1, 1, 0,0], fill="toself",opacity=0.3,fillcolor='orange',hoverinfo='skip',line=dict(width=0),mode='lines',showlegend=False),row=2,col=1)
        fig.update_layout(  # height=600, width=600,
            title_text="index %d time %d" % (sim_index, first_time_point),
            yaxis_range=[-83, -52],yaxis2_range=[0,1],xaxis_range=[max(first_time_point-margin,0),min(first_time_point + window_size+margin,v.shape[0])], legend_orientation="h")
        return fig

    def save(self):
        data = self.data.data_per_recording
        config_path = self.config.config_path
        is_validation = self.is_validation
        path = os.path.join(MODELS_DIR, *self.config.config_path)[:-len(".config")]
        with open(path + ".eval", 'wb') as pfile:
            pickle.dump((data, config_path, is_validation), pfile)
    @staticmethod
    def running_mse(v,v_p,mse_window_size):
        mse_window_size = min(v.shape[0], mse_window_size)
        running_mse = np.power(v - v_p,2)[np.newaxis,:]
        total_mse=np.mean(running_mse)
        if mse_window_size > 2:
            running_mse=uniform_filter1d(running_mse, size=mse_window_size, mode='constant')
            running_mse[:,:(mse_window_size // 2)] = 0
            running_mse[:,-(mse_window_size // 2 - 1):] = 0
        return running_mse,total_mse
    @staticmethod
    def load(path: str):
        if not path.endswith('.eval'):
            path += '.eval'
        if not MODELS_DIR in path:
            path = os.path.join(MODELS_DIR, path)
        with open(path, 'rb') as pfile:
            obj = pickle.load(pfile)
        config = configuration_factory.load_config_file(os.path.join(MODELS_DIR, *obj[1]))
        data = EvaluationData(obj[0])
        obj = ModelEvaluator(config, obj[2])
        obj.data = data
        return obj

    @staticmethod
    def load_data_generator(config, is_validation):
        train_files, valid_files, test_files = load_files_names()
        data_files = valid_files if is_validation else test_files
        validation_data_generator = SimulationDataGenerator(data_files, buffer_size_in_files=BUFFER_SIZE_IN_FILES_VALID,
                                                            batch_size=8,
                                                            window_size_ms=config.time_domain_shape,
                                                            file_load=config.train_file_load,
                                                            sample_ratio_to_shuffle=1,
                                                            # number_of_files=1,number_of_traces_from_file=2,# todo for debugging
                                                            ).eval()
        return validation_data_generator

    @staticmethod
    def build_and_save(config_path='', config=None, model=None):
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
    # model_name='AdamWshort_and_wide_1_NMDA_Tree_TCN__2022-02-06__15_47__ID_54572'
    # model_name = 'd_out_AdamW___2022-03-27__19_52__ID_52394'
    # model_name = 'd_out_AdamW___2022-03-27__19_52__ID_36974'
    # model_name = 'd_out_1_AdamW___2022-03-27__19_25__ID_12056'
    # model_name = 'd_out_1_AdamW___2022-03-27__19_25__ID_26687' # 0.975
    # model_name = 'd_out_5_AdamW___2022-04-09__20_08__ID_4486' # 0.97918
    # model_name = 'narrow_last_layer_3_AdamW___2022-06-28__12_07__ID_44880' # 0.0.96779
    # model_name = 'narrow_last_layer_3_AdamW___2022-06-28__12_07__ID_74982' # 0.97033
    # model_name = 'narrow_last_layer_4_AdamW___2022-07-04__11_05__ID_65095' # 0.96937
    # model_name = 'narrow_last_layer_4_AdamW___2022-07-04__11_05__ID_5433' # 0.96765
    # model_name = 'tempspace_AdamW___2022-07-18__15_54__ID_39553' # 0.97332
    # model_name = 'tempspace_AdamW___2022-07-18__15_54__ID_63486' # 0.97317
    # model_name = 'davids_AdamW___2022-07-18__12_52__ID_61236' # 0.98047
    model_name = 'davids_AdamW___2022-07-18__12_52__ID_70365' # 0.98120
    model_name = 'davids_leak_AdamW___2022-07-28__17_32__ID_42419' # 0.97761




    # model_name = 'd_out_5_AdamW___2022-04-09__20_08__ID_55904' #0.97762


    # model_name = 'glu_3_AdamW___2022-02-17__14_33__ID_2250'
    # model_name = 'glu_3_NAdam___2022-02-17__14_33__ID_60416'
    # ModelEvaluator.build_and_save(r"C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\models\NMDA\heavy_AdamW_NMDA_Tree_TCN__2022-01-27__17_58__ID_40048\heavy_AdamW_NMDA_Tree_TCN__2022-01-27__17_58__ID_40048")
    eval = ModelEvaluator.load(
        r"C:\Users\ninit\Documents\university\Idan_Lab\dendritic tree project\models\NMDA\%s\%s.eval" % (
        model_name, model_name))
    # eval.data.flatten_batch_dimensions()
    # eval.save()
    eval.display()
