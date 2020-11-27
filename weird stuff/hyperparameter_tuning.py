
if args.tied:
    emsizes = [10]
    nhids = [None]
else:
    emsizes = [10]
    nhids = [10]

dropouts = [0, 0.2, 0.5]
nonlins = ['tanh','relu','sigmoid']

epochs = 3
lr = args.lr


try:
    for emsize in emsizes:
        for nhid in nhids:
            for dropout in dropouts:
                for nonlin in nonlins:
                    if nhid is None:
                        nhid = emsize

                    best_val_loss = None
                    torch.manual_seed(args.seed)
                    args.model = 'FeedForward'
                    args.nonlin = nonlin
                    args.nhid = nhid
                    args.emsize =emsize
                    args.dropout = dropout
                    
                    # RUN  python main.py --model=FeedForward --epochs=5 --cuda --norder=8 --save=
                    #with configs
            

                    for epoch in range(1, epochs+1):
                        
                        train(shorter_data=True)
                        val_loss = evaluate(val_data[:100])
                        if not best_val_loss or val_loss < best_val_loss:
                            best_val_loss = val_loss
                        else:
                            lr /= 4.0
                    print("| emsize {:3d} | nhid {:3d} | dropout {:02.1f} | nonlin: {} |val loss {:02.4f} ".format(emsize, nhid, dropout, nonlin, best_val_loss))
                    # save to args.save_data 


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')