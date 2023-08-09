import util.dataset_util as ds_u

def misclassified_items(predicted, target):
    misclassified_rows = []

    for i, (row1, row2) in enumerate(zip(predicted, target)):
         if row1 != row2:
            misclassified_rows.append(i)


def finding_misclassifed_texts(ouput_file:dict):
    distilbert_output = ds_u.read_output_data('distilbert.pkl')
    minilm_output = ds_u.read_output_data('mini-lm.pkl')
    xlm_r_b_output = ds_u.read_output_data('xlm-r.pkl')

    if distilbert_output['task_id'] == 'AD':
        task_id = 'AD'
        distilbert_miscls = misclassified_items(distilbert_output['AD']['predicted'],distilbert_output['AD']['target'])
        minilm_miscls = misclassified_items(minilm_output['AD']['predicted'],minilm_output['AD']['target'])
        xlm_r_b_miscls =  misclassified_items(xlm_r_b_output['AD']['predicted'],xlm_r_b_output['AD']['target'])
        matching_items = set(distilbert_miscls) & set(minilm_miscls) & set(xlm_r_b_miscls)
        matching_items = list(matching_items)
        misclassified_texts = []
        misclassified_labels = []
        print(f"matching items are", matching_items)
        for id, item in enumerate(matching_items):
            misclassified_texts.append(distilbert_miscls['AD']['texts'][item])
            misclassified_labels.append(distilbert_miscls['AD']['targets'][item])

            print("Task id is: ", task_id)
            print(misclassified_texts[id]+' '+misclassified_labels[id])
        








