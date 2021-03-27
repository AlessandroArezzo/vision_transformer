# vision_transformer

Il progetto contiene un'implementazione in linguaggio Python dell'architettura Vision Transformer (ViT) proposta nel paper "An Image Is Worth 16x16 Words:
Transformers For Image Recognition At Scale", Anonymous, 2020.

<h2>Implementazione</h2>
Il progetto si articola su 3 file principali denominati ViT_model.py, hybrid_ViT_model.py e dataset.py.
<ul>
  <li><b>ViT_model.py:</b> contiene l'implementazione del modello ViT realizzata usando il framework Pytorch alla versione 1.4.0. 
    La classe che si occupa di definire l'architettura estende nello specifico nn.Module e nel farlo prevede l'esecuzione in sequenza di oggetti appartenenti alle classi EmbeddingLayer, TransformerEncoder ed MLPHead definite all'interno dello stesso file. <br>
    La prima di queste consente di partizionare l'immagine di input in patches, di proiettarne ogni loro rappresentazione flatten usando un layer lineare e di sommare ad ogni patch embedding il corrispondente vettore di positional encoding, una volta aggiunto il cls token.
    <br>
    TransformerEncoder, poi, consiste nell'esecuzione in cascata di singoli blocchi definiti dalla classe TransformerEncoderBlock, i quali implementano gli encoder elementari che caratterizzano l'architettura ViT. Per farlo questi fanno ricorso alle classi di MultiheadAttention, MLP, Norm e ResidualConnection che implementano il meccanismo di self-attention, la rete feed forward interna ad ogni encoder ed i moduli di normalizzazione e connessione residuale da eseguire rispettivamente prima e dopo ciascun elemento.<br>
    La classe MLPHead infine implementa un singolo layer lineare che rappresenta la rete fully connected atta a classificare le immagini una volta ricevuta la prima uscita del TransformerEncoder. </li>
    <li><b>hybrid_ViT_model.py:</b> fornisce l'implementazione dei modelli ViT ibridi. Questa, è definita per mezzo di una classe astratta HybridViT, la quale utilizza i moduli definiti in ViT_model.py fatta eccezione per il primo di essi, sostituito da un oggetto della classe HybridEmbeddingLayer per far fronte al fatto che in tal caso l'embedding non partiziona direttamente l'immagine di input ma la feature map estratte usando una qualche CNN. Tale rete da utilizzare a questo scopo deve essere definita implementando un metodo denominato get_CNN_backbone() all'interno di ciascuna classe concreta che estenda appunto la struttura astratta appena introdotta.<br>
      Nel progetto è stata nello specifico definita una classe denonimata Resnet18HybridViT che estrae le features dal quarto blocco di una rete resnet18, per la quale viene usata l'implementazione definita all'interno della libreria torchvision.</li>
     <li><b>dataset.py:</b> contiene la struttura che consente di definire i data loader dei tre datasets presi in considerazione per l'elaborato: CIFAR-10, CIFAR-100 ed Oxford-IIIT Pets. Per i primi due, all'interno del file è definita una classe denominata LoadTorchData, la quale utilizza il modulo interno alla libreria torchvision per scaricare i datasets stessi ed ottenerne i data loader relativi a training e test set. Se richiesto, i dati di train vengono suddivisi in modo da estrarne una certa percentuale ai fini della definizione di un validation set. Il dataset Oxford-IIIT Pets è implementato invece estendendo la classe CustomDataset, la quale definizione si è resa necessaria per tutti quei set di dati non inclusi in torchvision. Anche in questo caso, i dati di train vengono divisi all'occorrenza per ottenere una certa parte di immagini di validation.</li>
</ul>


<h2>Come usare il codice</h2>
Per addestrare i modelli, all'interno del progetto è incluso lo script <b>train_models.py</b>. Questo può nello specifico essere eseguito in tre differenti modalità settando il parametro --eval_type nei seguenti modi possibili:
<ul>
  <li>val: durante il training il modello viene valutato ad ogni epoca sui dati di validation. Al termine della procedura viene salvato l'andamento della loss function e dell'accuracy rilevate durante il processo sia sui dati di train che si quelli di validazione.</li>
  <li>test: durante il training il modello viene valutato ad ogni epoca sui dati di test. Al termine della procedura vengono salvati i risultati ottenuti comprendenti la top1 accuracy rilevata in un file csv.
  <li>both:  durante il training il modello viene valutato ad ogni epoca sia sui dati di validation che su quelli di test. Lo script genera in ouput i risultati di entrambe le modalità. </li>
</ul>
Lo script consente poi il settaggio dei seguenti parametri:
<ul>
  <li>--dataset_name: nome del dataset su cui addestrare i dati (CIFAR-10, CIFAR-100 o oxfordpets) </li>
  <li>--dataset_path: percorso in cui si trovano i dati. Nel caso di CIFAR-10 e CIFAR-100 questi vengono scaricati da torchvision nel percorso se non già presenti al suo interno. </li>
  <li>--output_root_path: percorso in cui salvare i dati generati in output. </li>
  <li>--n_classes: numero di classi delle immagini contenute nel dataset.</li>
  <li>--n_channels: numero di canali delle immagini contenute nel dataset.</li>
  <li>--val_ratio: percentuale nella quale dividere i dati di train per ottenere quelli di validation (se eval_type è settato su modalità val o both). </li>
  <li>--n_epochs: numero di epoche di train. </li>
  <li>--batch_size_train: batch size di train.</li>
  <li>--batch_size_test: batch size di test.</li>
  <li>--lr: learning rate di partenza.</li>
  <li>--n_cpu: numero di core da usare per il processo.</li>
  <li>--cuda: se settato il calcolo avviene su gpu.</li>
  <li>--CUDA_VISIBLE_DEVICES: definisce il bus id della CPU da utilizzare (se cuda è settato).</li>
  <li>--data_augmentation: se settato viene applicata data augmentation ai dati di train (crop e flip casuali).</li>
  <li>--weight_decay: weight decay da utilizzare.</li>
  <li>--optimizer: ottimizzatore da usare (adam o sgd).</li>
  <li>--model_type: modello da addestrare. I modelli addestrabili sono resnet18, ViT-XS e ViT-S (per la configurazione di questi ultimi si rimanda al report del progetto).</li>
  <li>--hybrid: se settato viene addestrato il modello ibrido della configurazione di ViT definita al parametro model_type.</li>
  <li>--patch_size: dimensione delle patches nelle quali partizionare le immagini.</li>
  <li>--image_size: dimensione alla quale effettuare il resize delle immagini del dataset.</li>
  <li>--dropout: indica probabilità del dropout.</li>
</ul>

<h2>Risultati</h2>
Se lo script train_models.py viene eseguito in modalità validation, questo salva i grafici dell'andamento di loss function ed accuracy rilevati sui dati di train e di validazione all'interno di una directory avente il nome del modello stesso e creata al percorso definito dal parametro output_root_path.
Se viceversa lo script è eseguito in modalità test, i risultati rilevati sul test set durante il training vengono salvati in un file csv denonimato models_results.csv anch'esso interno alla directory specificata dal parametro output_root_path e che conterrà un resoconto delle performance rilevate durante ogni training eseguito.
Inoltre, ogni modello viene salvato ad ogni epoca in un file .pth sempre all'interno della directory avente lo stesso nome del modello.<br>
Si noti di come il nome dei modelli ViT segua la nomenclatura riportata nel paper "An Image Is Worth 16x16 Words:
Transformers For Image Recognition At Scale", Anonymous, 2020. Ovvero il nome attribuito ad ogni Vision Tranformer è pari a quello della sua configurazione (ViT-XS, ViT-S) seguito da un numero che ne indica il patch size utilizzato.

<h2>Prerequisiti</h2>
Per eseguire il codice è necessario disporre della distribuzione di Python v.3 con le seguenti librerie installate:

<ul>
  <li>torch v.1.4.0</li>
  <li>torchvision v.0.5.0</li>
  <li>einops v.0.3.0</li>
  <li>matplotlib v.2.1.1</li>
  <li>pandas v.0.22.0</li>
</ul>

