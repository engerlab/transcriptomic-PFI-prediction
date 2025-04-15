import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from miRNA_pipe import miRNA_load_data
from clinical_pipe import clinical_load_data
from mRNA_pipe import mRNA_load_data
import plotly.graph_objects as go
import plotly.io as pio
from Mink import science_template
pio.templates['science'] = science_template
pio.templates.default = 'science'

def get_cohort_overlap():
    mRNA_id, mRNA_data, mRNA_gene_names, mRNA_matched_labels, mRNA_matched_times  = mRNA_load_data()
    miRNA_id, miRNA_data, miRNA_gene_names, miRNA_matched_labels, miRNA_matched_times  = miRNA_load_data()

    clinical_df = clinical_load_data()


    print(clinical_df.shape)
    print(mRNA_id.shape)
    print(miRNA_id.shape)


    mRNA_set = set(mRNA_id)
    miRNA_set = set(miRNA_id)
    clinical_set = set(clinical_df['case_id'].values)


    clin_m = set.intersection(mRNA_set, clinical_set)
    print(len(clin_m))

    clin_mi = set.intersection(miRNA_set, clinical_set)
    print(len(clin_mi))
    

    m_mi = set.intersection(miRNA_set, mRNA_set)
    print(len(m_mi))

    overlap = set.intersection(mRNA_set, miRNA_set, clinical_set)
    print(len(overlap))

def get_radiation_dose():

    labeled_data = clinical_load_data()
    name = 'radiation_therapy_YES'
    counts = labeled_data[name].value_counts()


    # mRNA_set = set(mRNA_id)
    # miRNA_set = set(miRNA_id)
    # clinical_set = set(clinical_df['case_id'].values)


    # print(counts)
    names = counts.index
    values = counts.values
    fig=go.Figure()
    fig.add_trace(go.Bar(x=names, y=values))

    if isinstance(names[0], (int, float)):
        # print(type(names[0]))
        mean = np.mean(names)
        print(np.min(names))
        print(np.max(names))
        median = np.median(names)
        print(median)

        fig.add_annotation(xref='paper', yref='paper', y=1, x=1, text=f"mean = {mean:.1f} median={median:.1f} mode={names[np.argmax(values)]}", xanchor='right', showarrow=False)
        
    fig.update_layout(yaxis_showgrid=True, width=1100, height=800, xaxis= dict(tickangle=-45, tickfont_size=10),
                        yaxis_title="counts", title=name,
                        margin=dict(t=70, b=70, r=70, l=70), font_size=27)

    print(labeled_data[name].value_counts())
    print(labeled_data[name].value_counts().sum()/len(labeled_data))
    print(len(labeled_data))
    fig.show()
if __name__ == '__main__':
    get_radiation_dose()
