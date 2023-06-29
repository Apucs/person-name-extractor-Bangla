from data import config
from build_dataloader import BuildDataLoader
import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support


def accuracy(corpus, preds, y):
    """
    Accuracy score based on given predictions and ground truths
    Args:
        corpus: Data corpus on which data loader, vocabulary and output tags has been built.
        preds: Predictions by model
        y: Ground truths

    Returns: Accuracy score, metrics(Precision, Recall, F1-score)

    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability

    non_pad_elements = (y != corpus.tag_pad_idx).nonzero()  # prepare masking for paddings
    # print("true tags:", y[non_pad_elements].squeeze(1).numpy())
    # print("predicted tags:", max_preds[non_pad_elements].squeeze(1).squeeze(1).numpy())
    metrics = precision_recall_fscore_support(y[non_pad_elements].squeeze(1).numpy(),
                                              max_preds[non_pad_elements].squeeze(1).squeeze(1).numpy(),
                                              average='weighted')

    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]), metrics


def evaluation(checkpoint_path, corpus, iterator):
    """
    Doing the evaluation on the given dataloader
    Args:
        checkpoint_path: Path to the trained model
        corpus: Data corpus on which data loader, vocabulary and output tags has been built.
        iterator: Dataloader on which evaluation would be done

    Returns: Evaluation results. (Loss, accuracy)

    """
    epoch_loss = 0
    epoch_acc = 0

    precision = 0
    recall = 0
    f1_score = 0

    model = torch.jit.load(checkpoint_path)

    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in iterator:
            text = batch.word

            true_tags = batch.tag

            pred_tags = model(text)

            pred_tags = pred_tags.view(-1, pred_tags.shape[-1])

            top_tags = pred_tags.argmax(-1)

            predicted_tags = [corpus.tag_field.vocab.itos[t.item()] for t in top_tags]

            true_tags = true_tags.view(-1)

            batch_loss = loss_fn(pred_tags, true_tags)
            batch_acc, result = accuracy(corpus, pred_tags, true_tags)
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
            precision += result[0]
            recall += result[1]
            f1_score += result[2]

    results = {
        "loss": epoch_loss / len(iterator),
        "accuracy": epoch_acc / len(iterator),
        "precision": precision / len(iterator),
        "recall": recall/len(iterator),
        "f1_score": f1_score/len(iterator)
    }
    return results


def main():
    checkpoint_path = config.CHECKPOINT3
    dataloader = BuildDataLoader(0)
    corpus = dataloader.corpus
    results = evaluation(checkpoint_path, corpus, corpus.test_iter)
    print(f"Test Loss: {results['loss']:.3f} |  Test Acc: {results['accuracy'] * 100:.2f}%")
    print(f"Precision: {results['precision']*100:.2f}% | Recall: {results['recall']*100:.2f}% | "
          f"F1_score: {results['f1_score']*100:.2f}%")


if __name__ == '__main__':
    main()
