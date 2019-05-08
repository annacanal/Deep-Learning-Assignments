function [RNN, sloss, iter, M, min_RNN, min_h, min_iter, min_loss] = MiniBatchGD(RNN, data, seq_length, K, m, eta,iter, M, ind_to_char, smooth_loss, min_loss)

    e = 1;
    %text_length = 200;
    text_length = 1000;
    sloss = [];
    while e <= length(data) - seq_length - 1
        X_batch = data(:, e:e + seq_length-1);
        Y_batch = data(:, e+1:e + seq_length);
        if e == 1
            hprev = zeros(m, 1);
        else
            hprev = h(:, end);
        end
        [loss, a, h, ~, p] = forward(RNN, X_batch, Y_batch, hprev, seq_length, K, m);
        [RNN, M] = backward(RNN, X_batch, Y_batch, a, h, p, seq_length, m, eta, M);

        if iter == 1 && e == 1
            smooth_loss = loss;
        end
        smooth_loss = 0.999*smooth_loss + 0.001*loss;
        if smooth_loss < min_loss
            min_RNN = RNN;
            min_h = hprev;
            min_iter = iter;
            min_loss = smooth_loss;
        end
        sloss = [sloss, smooth_loss];
        %every 100 update step, print out smooth loss
%         if iter == 1 || mod(iter, 100) == 0
%             disp(['iter = ' num2str(iter) ', smooth_loss = ' num2str(smooth_loss)]);
%         end

        %every 1000 update step, synthesize text
%         if iter == 1 || mod(iter, 10000) == 0
%             y = synthesize(RNN, hprev, data(:, 1), text_length, K);
%             c = [];
%             for i = 1:text_length
%                 c = [c ind_to_char(y(i))];
%             end
%             fprintf('\n');
%             disp(['iter = ' num2str(iter) ', smooth_loss = ' num2str(smooth_loss)]);
%             disp(c);
%         end

        iter = iter + 1;
        e = e + seq_length;
    end

end