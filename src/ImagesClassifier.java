import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.ClassificationViaClustering;
import weka.classifiers.rules.NNge;
import weka.classifiers.trees.DecisionStump;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class ImagesClassifier {


    private static final String DIGITOS = "digitos";
    private static final String LETRAS = "letras";
    private static final String DIGITOS_LETRAS = "digitos_letras";
    private static final String SEM_CARACTERES = "sem_caracteres";
    private static final String[] classe_names = {DIGITOS, LETRAS, DIGITOS_LETRAS, SEM_CARACTERES};
    private static final double LUMINANCE_RED = 0.299D;
    private static final double LUMINANCE_GREEN = 0.587D;
    private static final double LUMINANCE_BLUE = 0.114;
    private static final int HIST_WIDTH = 256;    
    private static final int HIST_HEIGHT = 100;
    private static final int INDEX = HIST_WIDTH;
    private static final int CAPACITY = 257;
    private static FastVector wekaAtributos = new FastVector(CAPACITY);
    
    public static void main(String[] args) throws Exception {
    	
    	Boolean verbose = Boolean.valueOf(args[0]);
    	String classificadorEscolhido = args[1];
    	String caminhoPastaTreinamento=args[2];
    	String caminhoPastaTeste = args[3];
    	
    	
    	//Mapa com os classificadores que serao usados
    	Map<String, Classifier> classificadores = new HashMap<String, Classifier>() {
			private static final long serialVersionUID = 1L;

		{
            put("NaiveBayes", new NaiveBayes());
            put("DecisionStump", new DecisionStump());
            put("MultilayerPerceptron", new MultilayerPerceptron());
            put("NNge", new NNge());
            put("ClassificationViaClustering", new ClassificationViaClustering());
        }};
    	
        //Preenche atributos do header
        for (int i = 0; i < HIST_WIDTH ; i++) {
            Attribute attr = new Attribute("numeric" + i);
            wekaAtributos.addElement(attr);
        }
        
        //Vetor de classes, define as classes as quais sera feita a classificacao. 
        //As imagens devem sem classificadas de acordo com essas classes, elas devem se enquadrar em uma dessas classes.

        FastVector classes= new FastVector(4);
        for (int i = 0; i < classe_names.length; i++) {
			classes.addElement(classe_names[i]);
		}
        
        //O conjunto dessas classes ira compor um atributo, todo atributo deve ser adicionado no vetor de atributos principal wekaAtributos.
        Attribute attr = new Attribute("classes",classes);
        wekaAtributos.addElement(attr);
        
        //Classificador usado, define a tecnica de classificacao que vai ser usada. 
        Classifier cModel = classificadores.get(classificadorEscolhido);
        if(cModel == null){
        	System.err.println("Classificador " + classificadorEscolhido + " nao encontrado\n"
        			+ "execute ./classificar -h para lista de classificadores.");
        	System.exit(1);
        }
        
        //Conjunto de treino, vai servir como base de comparacao para classificar as imagens.
        //O conjunto de treino deve ser adicionado no classificador para ser avaliado (classificado).        
        List<ImgItem> trainingImages = setUpImages(caminhoPastaTreinamento);
        Instances conjuntoTreinamento = createAnalysisSet("TrainingSet", trainingImages);
        
        //Conjunto de teste, vai servir como base de comparacao para classificar as imagens.
        //O conjunto de treino deve ser adicionado no classificador para ser avaliado (classificado).         
        List<ImgItem> testImages = setUpImages(caminhoPastaTeste);
        Instances conjuntoTest = createAnalysisSet("TestSet", testImages);
        
        if(conjuntoTreinamento.numInstances() > 0 && conjuntoTest.numInstances() > 0){
	        cModel.buildClassifier(conjuntoTreinamento);
	        
	        cModel.buildClassifier(conjuntoTest);
	        
			//Objeto responsavel por aplicar a classificacao em um conjunto de testes, usando como base um conjunto de treinamento.
	        Evaluation eTest = new Evaluation(conjuntoTreinamento);
	        eTest.evaluateModel(cModel,conjuntoTest);
	        
	        if(verbose){
	        	FastVector predictions = eTest.predictions();
	        	for (int i = 0; i < predictions.size(); i++) {
					System.out.println(
							testImages.get(i).getImgName() + ": " +
					classe_names[(int) ((Prediction) predictions.elementAt(i)).predicted()]);
				}
	        }
	
	        System.out.printf("precision: %.2f\n", eTest.weightedPrecision());
	        System.out.printf("recall: %.2f\n", eTest.weightedRecall());
	        System.out.printf("f-measure: %.2f\n", eTest.weightedFMeasure());
        }
    }
    
    /**
     * Abre imagens e constroi objetos com histogramas
     * @param caminho caminho de pastas digitos, digitos_letras...etc
     * @return lista de instancias prontas para serem usadas
     */
    private static List<ImgItem> setUpImages(String caminho) {
    	List<ImgItem> instanceList = new ArrayList<ImgItem>();
    	int tamanhoCaminho = caminho.length() -1;
		for (int i = 0; i < classe_names.length; i++) {
	        final File pastaImagens = new File(caminho.charAt(tamanhoCaminho) == '/'
	        		? caminho+classe_names[i] : caminho + "/" + classe_names[i]);
	        List<File> imagens;
			try {
				imagens = listFilesForFolder(pastaImagens);
				for (File file : imagens) {
					try {
						double[] histograma = buildHistogram(file);
						if(histograma.length == INDEX){
							instanceList.add(new ImgItem(file.getName(), classe_names[i], histograma));
						}
					} catch (IOException e) {
					}
				}
			} catch (IOException e1) {
				System.err.println(e1.getMessage());
			}	        
		}
		return instanceList;
	}

	/**
     * Cria uma instancia arff para uma dada imagem
     * 
     * @param FastVector Vetor de atributos wekaAtributos
     * @param histograma
     * @param classe classe da imagem
     */
    private static Instances createAnalysisSet(String setName, List<ImgItem> imagesItem) {
    	Instances analysisSet = new Instances(setName, wekaAtributos, imagesItem.size());
        analysisSet.setClassIndex(INDEX);
        for (ImgItem img : imagesItem) {
        	Instance imageInstance = new Instance(CAPACITY);
			for (int i = 0; i < img.getImgHistogram().length; i++) {
				imageInstance.setValue((Attribute) wekaAtributos.elementAt(i), img.getImgHistogram()[i]);				
			}
			imageInstance.setValue((Attribute) wekaAtributos.elementAt(INDEX), img.getImgClass());
			analysisSet.add(imageInstance);
		}
        
        return analysisSet;
    }
    
    /**
     * Retorna uma lista de arquivos que estao contidos dentro de uma pasta
     * 
     * @param File pasta
     * @return List<File> lista de arquivos da pasta
     * @throws IOException 
     */
    private static List<File> listFilesForFolder(final File folder) throws IOException {
    	if(folder.listFiles() == null)
    		throw new IOException(folder.getAbsolutePath() + " nao encontrado");
    	List<File> arquivos = new ArrayList<File>();
        for (final File fileEntry : folder.listFiles()) {
            if(fileEntry.getName().endsWith(".jpg")){
            	arquivos.add(fileEntry);
            }
        }
        return arquivos;
    }

//////////////// helper code ////////////////////////

    /**
     * Parses pixels out of an image file, converts the RGB values to
     * its equivalent grayscale value (0-255), then constructs a
     * histogram of the percentage of counts of grayscale values.
     *
     * @param infile - the image file.
     * @return - a histogram of grayscale percentage counts.
     * 
     * Codigo retirado de http://sujitpal.blogspot.com.br/2012/04/image-classification-photo-or-drawing.html
     * @throws IOException 
     */
    protected static double[] buildHistogram(File infile) throws IOException{
        BufferedImage input = ImageIO.read(infile);
        int width = input.getWidth();
        int height = input.getHeight();
        List<Integer> graylevels = new ArrayList<Integer>();
        double maxWidth = 0.0D;
        double maxHeight = 0.0D;
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < height; col++) {
                Color c = new Color(input.getRGB(row, col));
                int graylevel = (int) (LUMINANCE_RED * c.getRed() +
                        LUMINANCE_GREEN * c.getGreen() +
                        LUMINANCE_BLUE * c.getBlue());
                graylevels.add(graylevel);
                maxHeight++;
                if (graylevel > maxWidth) {
                    maxWidth = graylevel;
                }
            }
        }
        double[] histogram = new double[HIST_WIDTH];
        for (Integer graylevel : (new HashSet<Integer>(graylevels))) {
            int idx = graylevel;
            histogram[idx] +=
                    Collections.frequency(graylevels, graylevel) * HIST_HEIGHT / maxHeight;
        }
        return histogram;
    }

    // Classe auxiliar para armazenar informacoes necessarias sobre imagem
    static class ImgItem{
    	
    	private String imgName;
    	private String imgClass;
    	private double[] imgHistogram;
    	
    	ImgItem(String imgName, String imgClass, double[] imgHistogram){
    		this.imgName = imgName;
    		this.imgClass = imgClass;
    		this.imgHistogram = imgHistogram;
    	}

		public String getImgName() {
			return imgName;
		}

		public String getImgClass() {
			return imgClass;
		}

		public double[] getImgHistogram() {
			return imgHistogram;
		}
    }

}


