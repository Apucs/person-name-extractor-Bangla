import torch
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt


class NER(object):

    def __init__(self, model, data, optimizer_cls, loss_fn_cls):
        """
        Initializing the extraction model for training
        Args:
            model: Initialized model
            data: Data corpus on which data loader, vocabulary and output tags has been built.
            optimizer_cls: Optimization technique that will be used
            loss_fn_cls: Loss function which will be used while training
        """
        self.model = model
        self.data = data
        self.optimizer = optimizer_cls(model.parameters())
        self.loss_fn = loss_fn_cls(ignore_index=self.data.tag_pad_idx)

    @staticmethod
    def epoch_time(start_time, end_time):
        """
        Time that is needed for each epoch
        Args:
            start_time: Starting time of the particular epoch
            end_time: Ending time of the epoch

        Returns:
            elapsed_mins: Elapsed time in minutes
            elapsed_secs: Elapsed time in seconds

        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    @staticmethod
    def plot_graph(x_acc, x_loss, epoch, plot_type):
        """
        Plot and saving the training and validation graph based on the plot type
        Args:
            x_acc: List of accuracy scores to plot the graph
            x_loss: List of loss scores to plot the graph
            epoch: No of epochs
            plot_type: str
                Whether it's a training plot or testing plot

        Returns: None

        """
        if plot_type == "train":
            fig, ax = plt.subplots()
            ax.plot(epoch, x_acc, label="accuracy")
            ax.plot(epoch, x_loss, label="loss")
            ax.set_xlabel('No of epoch(train)')
            ax.legend()
            plt.savefig("output/train.png")

        elif plot_type == "val":
            fig, ax1 = plt.subplots()
            ax1.plot(epoch, x_acc, label="accuracy")
            ax1.plot(epoch, x_loss, label="loss")
            ax1.set_xlabel('No of epoch(validation)')
            ax1.legend()
            plt.savefig("output/validation.png")

    def accuracy(self, preds, y):
        """
        Accuracy score based on given predictions and ground truths
        Args:
            preds: Predictions by model
            y: Ground truths

        Returns: Accuracy score

        """
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        non_pad_elements = (y != self.data.tag_pad_idx).nonzero()  # prepare masking for paddings
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

    def epoch(self):
        """
        Training block for one epoch
        Returns:
            epoch_loss = training loss for one iteration
            epoch_acc = training accuracy for one iteration

        """
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for batch in tqdm(self.data.train_iter):
            # text = [sent len, batch size]
            text = batch.word
            # tags = [sent len, batch size]
            true_tags = batch.tag
            self.optimizer.zero_grad()
            pred_tags = self.model(text)
            # flatten pred_tags to [sent len, batch size, output dim]
            pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
            # flatten true_tags to [sent len * batch size]
            true_tags = true_tags.view(-1)

            if true_tags.shape[0] != pred_tags.shape[0]:
                continue

            else:
                batch_loss = self.loss_fn(pred_tags, true_tags)
                batch_acc = self.accuracy(pred_tags, true_tags)
                batch_loss.backward()
                self.optimizer.step()
                epoch_loss += batch_loss.item()
                epoch_acc += batch_acc.item()
        return epoch_loss / len(self.data.train_iter), epoch_acc / len(self.data.train_iter)

    def evaluate(self, iterator):
        """
        Evaluation function by given the test dataloader
        Args:
            iterator: Dataloader on which evaluation would be done

        Returns: Evaluation results. (Loss, accuracy)

        """
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                # print(batch)
                text = batch.word
                true_tags = batch.tag
                pred_tags = self.model(text)
                pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
                true_tags = true_tags.view(-1)
                batch_loss = self.loss_fn(pred_tags, true_tags)
                batch_acc = self.accuracy(pred_tags, true_tags)
                epoch_loss += batch_loss.item()
                epoch_acc += batch_acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    # main training sequence
    def train(self, n_epochs):
        """
        Start the training
        Args:
            n_epochs: Total number of epoch for the training

        Returns: None

        """

        best_acc = 0
        summary = []
        t_loss = []
        t_acc = []
        v_loss = []
        v_acc = []
        ep = []

        for i, epoch in enumerate(tqdm(range(n_epochs))):
            start_time = time.time()
            train_loss, train_acc = self.epoch()
            end_time = time.time()
            epoch_mins, epoch_secs = NER.epoch_time(start_time, end_time)
            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")

            test_loss, test_acc = self.evaluate(self.data.test_iter)
            print(f"\tTest Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%")
            summary.append([train_acc * 100, test_acc * 100, train_loss, test_loss])
            t_acc.append(train_acc)
            t_loss.append(train_loss)
            v_acc.append(test_acc)
            v_loss.append(test_loss)
            ep.append(i + 1)

            if test_acc > best_acc:
                best_acc = test_acc

                model_scripted = torch.jit.script(self.model)  # Export to TorchScript
                model_scripted.save('checkpoint/model_scripted.pt')  # Save

        df = pd.DataFrame(summary, columns=["train accuracy", "test accuracy", "train loss", "test loss"])

        df.to_csv('output/result.csv', index=False)
        print(df.head(5))

        self.plot_graph(t_acc, t_loss, ep, "train")
        self.plot_graph(v_acc, v_loss, ep, "test")

        model_last = torch.jit.script(self.model)  # Export to TorchScript
        model_last.save('checkpoint/model_last.pt')

        torch.save(self.model.state_dict(), "checkpoint/checkpoint.pth")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, "checkpoint/checkpoint_saved.pth")
