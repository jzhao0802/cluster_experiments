p0=pred_benchmark$data$prob.1
p1=clustered_preds_obj[[1]]$data$prob.1
p2=clustered_preds_obj[[2]]$data$prob.1
p3=clustered_preds_obj[[3]]$data$prob.1
df = data.frame(p0,p1,p2,p3)
colnames(df)=c("b", "c1", "c2", "c3")
ggplot(stack(df),aes(x=values, y=..scaled..,fill=ind)) + geom_density(alpha=0.5)
