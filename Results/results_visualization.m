% order of columns: loss, acc1, acc5

training = load('training-metris.csv');
validation = load('validation-metrics.csv');

figure(1)
subplot(3,1,1)
plot(training(:,1), training(:,2), '-r', validation(:,1), validation(:,2), '-g');
legend('Training Loss', 'Validation Loss', 'Location','southwest');
savefig('Loss.fig')

subplot(3,1,2)
plot(training(:,1), training(:,3), '-r', validation(:,1), validation(:,3), '-g');
ylim([0,1]);
legend('Training Acc1', 'Validation Acc1', 'Location','northwest');

subplot(3,1,3)
plot(training(:,1), training(:,4), '-r', validation(:,1), validation(:,4), '-g');
ylim([0,1]);
legend('Training Acc5', 'Validation Acc5', 'Location','northwest');

saveas(1, 'results2.png','png');
