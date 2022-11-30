
This is a fork of [gluonts-hierarchical-ICML-2021](https://github.com/rshyamsundar/gluonts-hierarchical-ICML-2021) 

## Setup
Download this code and rename the directory as 'gluonts-hierarchical-ICML-2021' then run following commands.
```
pip install --upgrade mxnet
cd gluonts-hierarchical-ICML-2021
pip install -e .
```

## Running

```
python experiments/run_experiment_with_best_hps.py --dataset dataset --method DeepVAR
```
where dataset is one of `{labour, traffic, tourism, wiki}` .

This will run the Our modified DeepVAR method with coherent loss function 5 times on the selected dataset with the best hyperparameters.

Results:


<table>
 <thead>
  <tr>
   <th>RESULTS ON ALL LEVELS</th>
   <th></th>
   <th></th>
   <th></th>
   <th></th>
   <th></th>
   <th></th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>Dataset</td>
   <td>       Level</td>
   <td>Our Model</td>
   <td>Hier-E2E</td>
   <td>DeepVAR</td>
   <td>DeepVAR+</td>
   <td>Best of Competing Methods</td>
  </tr>
  <tr>
   <td>Labour</td>
   <td>1</td>
   <td>*</td>
   <td>0.0311+/-0.0120</td>
   <td>0.0352+/-0.0079</td>
   <td>0.0416+/-0.0094</td>
   <td>0.0406+/-0.0002 (PERMBU-MINT)</td>
  </tr>
  <tr>
   <td></td>
   <td>2</td>
   <td>*</td>
   <td>0.0336+/-0.0089</td>
   <td>0.0374+/-0.0051</td>
   <td>0.0437+/-0.0078</td>
   <td>0.0389+/-0.0002 PERMBU-MINT)</td>
  </tr>
  <tr>
   <td></td>
   <td>3</td>
   <td>*</td>
   <td>0.0336+/-0.0082</td>
   <td>0.0383+/-0.0038</td>
   <td>0.0432+/-0.0076</td>
   <td>0.0382+/-0.0002 (PERMBU-MINT)</td>
  </tr>
  <tr>
   <td></td>
   <td>4</td>
   <td>*</td>
   <td>0.0378+/-0.0060</td>
   <td>0.0417+/-0.0038</td>
   <td>0.0448+/-0.0066</td>
   <td>0.0397+/-0.0003 (PERMBU-MINT)</td>
  </tr>
  <tr>
   <td>Traffic</td>
   <td>1</td>
   <td>0.0106+/-0.0028*</td>
   <td>0.0184+/-0.0091</td>
   <td>0.0225+/-0.0109</td>
   <td>0.0250+/-0.0082</td>
   <td>0.0087(ARIMA-ERM)</td>
  </tr>
  <tr>
   <td></td>
   <td>2</td>
   <td>0.0144+/-0.0002*</td>
   <td>0.0181+/-6.0086</td>
   <td>0.0204+/-0.0044</td>
   <td>0.0244+/-0.0063</td>
   <td>0.0112(ARIMA-ERM)</td>
  </tr>
  <tr>
   <td></td>
   <td>3</td>
   <td>0.0145+/-0.0005*</td>
   <td>0.0223+/-0.0072</td>
   <td>0.0190+/-0.0031</td>
   <td>0.0259+/-0.0054</td>
   <td>0.0255 (ARIMA-ERM)</td>
  </tr>
  <tr>
   <td></td>
   <td>4</td>
   <td>0.0967+/-0.0003*</td>
   <td>0.0914+/-0.0024</td>
   <td>0.0982+/-0.0012</td>
   <td>0.0982+/-0.0017</td>
   <td>0.1410 (ARIMA-ERM)</td>
  </tr>
  <tr>
   <td>Tourism</td>
   <td>1</td>
   <td>0.0622 +/- 0.0131</td>
   <td>0.0402+/-0.0040</td>
   <td>0.0519+/-0.0057</td>
   <td>0.0508+/-0.0085</td>
   <td>0.0472+/-0.0012 (PERMBU-MINT)</td>
  </tr>
  <tr>
   <td></td>
   <td>2</td>
   <td>0.0929 +/- 0.0072</td>
   <td>0.0658+/-6.0084</td>
   <td>0.0755+/-0.0011</td>
   <td>0.0750+/-00066</td>
   <td>&quot;0.0605+/-0.0006 (PERMBU-MINT)</td>
  </tr>
  <tr>
   <td>0.0605+/-0.0006 (PERMBU-MINT)&quot;</td>
  </tr>
  <tr>
   <td></td>
   <td>3</td>
   <td> 0.1287 +/- 0.0073</td>
   <td>0.1053+/-0.0053</td>
   <td>0.1134+/-0.0049</td>
   <td>0.1180+/-0.0053</td>
   <td>0.0903+/-0.0006 (PERMBU-MINT)</td>
  </tr>
  <tr>
   <td></td>
   <td>4</td>
   <td>0.1525 +/- 0.0057</td>
   <td>0.1223+/-0.0039</td>
   <td>0.1294+/-0.0060</td>
   <td>0. 1393+/-0.0048</td>
   <td>0.1106+/-0.0005 (PERMBU-MINT)</td>
  </tr>
  <tr>
   <td>Wiki</td>
   <td>1</td>
   <td>0.0857 +/- 0.0248</td>
   <td>0.0419+/-0.0285</td>
   <td>0.0905+/-0.0323</td>
   <td>0.0755+/-0.0165</td>
   <td>0.1558 (ETS-ERM)</td>
  </tr>
  <tr>
   <td></td>
   <td>2</td>
   <td>0.1329 +/- 0.0164</td>
   <td>0.1045+/-0.0151</td>
   <td> 0.1418+/-0.0249</td>
   <td>0.1289+/-010171</td>
   <td>0.1614(ETS-ERM)</td>
  </tr>

  <tr>
   <td></td>
   <td>3</td>
   <td>0.2424 +/- 0.0214</td>
   <td>0.2292+/-0.0108</td>
   <td>0.2597+/-0.0150</td>
   <td>0.2583+/-0.0281</td>
   <td>0.2010(ETS-ERM)</td>
  </tr>
  <tr>
   <td></td>
   <td>4</td>
   <td>0.2802 +/- 0.0228</td>
   <td>0.2716+/-0.0091</td>
   <td>0.2886+/-0.0112</td>
   <td>0.3108+/-0.0298</td>
   <td>0.2399 (ETS-ERM)</td>
  </tr>
  <tr>
   <td></td>
   <td>5</td>
   <td>0.4030 +/- 0.0208</td>
   <td>0.3720+/-0.0150</td>
   <td>0.3664+/-0.0068</td>
   <td>0.4460+/-0.0271</td>
   <td>0.3507 (ETS-ERM)</td>
  </tr>
  <tr>
   <td>* Training or in Queue</td>
   <td></td>
   <td></td>
   <td></td>
   <td></td>
   <td></td>
   <td></td>
  </tr>
 </tbody>
</table>

