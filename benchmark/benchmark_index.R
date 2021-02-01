#' @param data: cells * features matrix
#' @param cell_type: named vector
#' @param batch: named vector
#' @param k: number of NNs to classify negative cells

options(stringsAsFactors = FALSE)
require(RcppAnnoy)
require(ggplot2)
require(grid)
require(ggpubr)

color_used <- c(
  "cyan4",      "skyblue3",   "darkolivegreen3",   "lightpink",   "darkmagenta",   "brown",   "blueviolet", "bisque4",  "deeppink3",       "darkkhaki",      
  "dodgerblue4",     "goldenrod4",            "gainsboro",       "firebrick4",      "cadetblue3",
  "greenyellow",     "gray6",           "coral2",                     "yellow4",         
         "darkgoldenrod3",  "navy",            "deepskyblue3","antiquewhite3"
)

benchmark_idx = function(data, cell_type, batch, k = 20, k2 = 100, pos_thr = 1, 
                         plot = TRUE, prefix = '', 
                         color_batch = NULL, 
                         color_cell_type = NULL, 
                         color_index = NULL, 
                         point_size = 1){
    bench_list = list()
    
    falseRates = c()
    tmp_c = table(cell_type)
    
    tmp_b = table(cell_type, batch)
    unique_batch = unique(batch)
    
    
    
    vector_size = dim(data)[2]
    a <- new(AnnoyEuclidean, vector_size)
    a$setSeed(1)
    for(i in 1:dim(data)[1]){
        a$addItem(i-1, data[i,])
    }
    a$build(50)
    
    ks = c()
    for(i in 1:dim(data)[1]){
        k_ = min(k, tmp_c[cell_type[i]])
        ks = c(ks, k_)
        NNs = a$getNNsByItem(i-1, k_+1)[2:(k_+1)] + 1
        
        c_NN = cell_type[NNs]
        false_rate = sum(c_NN != cell_type[i]) / k_
        falseRates = c(falseRates, false_rate)
    }
    
    bench_list[['falseRates']] = falseRates
    bench_list[['NegativeCells']] = falseRates >= 0.5
    bench_list[['NumberOfNegativeCells']] = sum(falseRates >= 0.5)
    message('Number of negative cells: ', sum(falseRates >= 0.5))
    
    
    ######### discriminate TRUE POSITIVE CELLS
    truePositive = c()
    
    tmp_b_prob = t(apply(tmp_b, 1, function(x) x/sum(x)))
                       
    for(i in 1:dim(data)[1]){
        cur = Inf
        if(falseRates[i] < 0.5){
            k_2 = min(k2, tmp_c[cell_type[i]])
            NNs_ = a$getNNsByItem(i-1, k_2) + 1
            NNs_c = cell_type[i]
            NNs_i = NNs_c == cell_type[i]
            NNs_ = NNs_[NNs_i]
            k_2 = length(NNs_)
            
            NNs_b = batch[NNs_]
            
            b_prob_ = tmp_b_prob[cell_type[i],]
            
            max_b = 0
            for(b in unique_batch){
                if(b_prob_[b] > 0 & b_prob_[b] < 1){
                    p_b = sum(NNs_b == b)
                    sd_b = abs(p_b - k_2 * b_prob_[b]) / sqrt(k_2 * b_prob_[b] * (1-b_prob_[b]))
                    max_b = max(max_b, sd_b)
                }
            }
            cur = max_b
        }
        
        truePositive = c(truePositive, cur)
    }
    
    bench_list[['truePositiveRate']] = truePositive
    bench_list[['truePositiveCells']] = truePositive <= pos_thr 
    bench_list[['NumberOfTruePositiveCells']] = sum(truePositive <= pos_thr)
    message('Number of true positive cells: ', sum(truePositive <= pos_thr))
           
    labels = rep('Others', length(batch))
    labels[falseRates >= 0.5] = 'Negative cells'
    labels[truePositive <= pos_thr] = 'True positive cells'
    bench_list[['labels']] = labels
    
    if(plot){
        n = length(batch)
        ord = sample(n,n,replace=F)
        
        labels = rep('Others', length(ord))
        labels[falseRates[ord] >= 0.5] = 'Negative cells'
        labels[truePositive[ord] <= pos_thr] = 'True positive cells'
        df = data.frame(
            x = data[ord, 1],
            y = data[ord, 2],
            labels = labels
        )
        
        if(is.null(color_index)){
            color_index = color_used[c(1,9,19)]
        }
        #### plot cell type
        pdf(paste0(prefix, '_index.pdf'), height = 6, width = 8)
        p = df %>% ggplot() + geom_point(aes(x=x, y=y, color=labels), size = point_size, alpha = 1) + theme_pubr() +
            scale_color_manual(values = color_index, name = 'scBM') +
            xlab('UMAP 1') + ylab('UMAP 2') + theme(legend.position = 'right') +
            guides(colour = guide_legend(override.aes = list(size=5))) +
            scale_size_manual(values = .1)
        print(p)
        dev.off()
        
        pdf(paste0(prefix, 'true_positive_cells.pdf'), height = 6, width = 8)
        p = ggplot() + geom_point(aes(x=data[ord,1], y=data[ord,2], color=truePositive[ord] <= pos_thr), size = point_size, alpha = 1) + theme_pubr() +
            scale_color_manual(values = c('red', 'blue'), name = 'True Positive') +
            xlab('UMAP 1') + ylab('UMAP 2') + theme(legend.position = 'right') +
            guides(colour = guide_legend(override.aes = list(size=5))) +
            scale_size_manual(values = .1)
        print(p)
        dev.off()
        
        ### Plot cell type
        if(is.null(color_cell_type)){
            no_of_markers <- length(unique(cell_type))
            if (no_of_markers > length(color_used)) {
                color_cell_type <- colorRampPalette(brewer.pal(9, "Set1"))(no_of_markers+1)
            }
            else {
                color_cell_type <- color_used
            }
        }
        pdf(paste0(prefix,'cell_types.pdf'), height = 6, width = 8)
        p = ggplot() + geom_point(aes(x=data[ord,1], y=data[ord,2], color=cell_type[ord]), size = point_size, alpha = 1) + theme_pubr() +
            scale_color_manual(values = color_cell_type, name = 'Cell type') +
            xlab('UMAP 1') + ylab('UMAP 2') + theme(legend.position = 'right') +
            guides(colour = guide_legend(override.aes = list(size=5))) +
            scale_size_manual(values = .1)
        print(p)
        dev.off()
    
#          theme(
#                 axis.text.y = element_blank(),
#                 axis.text.x = element_blank(),
#                 legend.position = 'right',
#                 axis.ticks.x=element_blank(),
#                 axis.ticks.y=element_blank()
#             )
        ### Plot cell type
        if(is.null(color_batch)){
            color_batch = rev(color_used)
        }
        pdf(paste0(prefix,'batches.pdf'), height = 6, width = 8)
        
        p = ggplot() + geom_point(aes(x=data[ord,1], y=data[ord,2], color=batch[ord]), size = point_size, alpha = 1) + theme_pubr() +
            scale_color_manual(values = color_batch, name = 'Batch') +
            xlab('UMAP 1') + ylab('UMAP 2') + theme(legend.position = 'right') +
            guides(colour = guide_legend(override.aes = list(size=5))) +
            scale_size_manual(values = .1)
        print(p)
        dev.off()
    }
    
    return(bench_list)
}