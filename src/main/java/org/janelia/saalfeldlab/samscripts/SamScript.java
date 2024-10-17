package org.janelia.saalfeldlab.samscripts;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.Point;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.labeling.ConnectedComponents;
import net.imglib2.converter.Converters;
import net.imglib2.histogram.Real1dBinMapper;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.IntArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.logic.BoolType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.apache.http.HttpException;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class SamScript {

	public static final int HTTP_SUCCESS = 200;
	public static final int HTTP_CANCELLED = 499;

	public static final String EMBEDDING_REQUEST_ENDPOINT = "embedded_model";
	public static final String COMPRESS_ENCODING_PARAMETER = "encoding=compress";
	public static final String DEFAULT_SERVICE_URL = "https://samservice.janelia.org/";
	public static final String DEFAULT_MODEL_LOCATION = "src/main/resources/sam/sam_vit_h_4b8939.onnx";
	public static final int DEFAULT_RESPONSE_TIMEOUT = 10 * 1000;

	public static final RequestConfig defaultRequestConfig = RequestConfig.custom()
			.setConnectionRequestTimeout(DEFAULT_RESPONSE_TIMEOUT)
			.setSocketTimeout(10 * DEFAULT_RESPONSE_TIMEOUT)
			.setConnectTimeout(DEFAULT_RESPONSE_TIMEOUT)
			.build();

	public static final long LOW_RES_MASK_DIM = 256L;

	/* 4D low-res mask, expected to be 256x256 */
	public static final FloatBuffer noMaskBuffer = allocateDirectFloatBuffer((int)(LOW_RES_MASK_DIM * LOW_RES_MASK_DIM));
	public static final long[] maskShape = new long[]{1, 1, LOW_RES_MASK_DIM, LOW_RES_MASK_DIM};
	public static final FloatBuffer hasNoMaskInput = allocateDirectFloatBuffer(1);
	public static final long[] hasMaskFlagShape = new long[]{1};

	public static final String IMAGE_EMBEDDINGS = "image_embeddings";
	public static final String ORIG_IM_SIZE = "orig_im_size";
	public static final String POINT_COORDS = "point_coords";
	public static final String POINT_LABELS = "point_labels";
	public static final String MASK_INPUT = "mask_input";
	public static final String HAS_MASK_INPUT = "has_mask_input";
	public static final String MASKS = "masks";

	public static final String IOU_PREDICTIONS = "iou_predictions";
	public static final String LOW_RES_MASKS = "low_res_masks";

	public static BufferedImage toBufferedImage(Image img) {

		if (img instanceof BufferedImage) {
			return (BufferedImage)img;
		}

		BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);

		Graphics2D bGr = bimage.createGraphics();
		bGr.drawImage(img, 0, 0, null);
		bGr.dispose();

		return bimage;
	}

	public static <T extends NumericType<T>> byte[] raiToPng(RandomAccessibleInterval<T> img) throws IOException {

		ByteArrayOutputStream predictionImagePngOutputStream = new ByteArrayOutputStream();

		BufferedImage image = toBufferedImage(ImageJFunctions.wrap(img, "asPng").getImage());
		ImageIO.write(image, "png", predictionImagePngOutputStream);
		return predictionImagePngOutputStream.toByteArray();
	}

	public static OrtSession createOrtSession(OrtEnvironment env) throws IOException, OrtException {

		byte[] modelArray = Files.readAllBytes(Paths.get(DEFAULT_MODEL_LOCATION));
		return env.createSession(modelArray);
	}

	public static CloseableHttpClient createSamServiceClient() {

		return HttpClientBuilder.create().useSystemProperties().setDefaultRequestConfig(defaultRequestConfig).build();
	}

	public static HttpPost createSamEmbeddingPostRequest(byte[] png, String sessionId, boolean compressEncoding) {

		MultipartEntityBuilder entityBuilder = MultipartEntityBuilder.create();
		entityBuilder.addBinaryBody("image", png, ContentType.APPLICATION_OCTET_STREAM, "null");

		if (sessionId != null) {
			entityBuilder.addTextBody("session_id", sessionId);
			entityBuilder.addTextBody("cancel_pending", "true");
		}

		String url = compressEncoding ?
				String.format("%s/%s?%s", DEFAULT_SERVICE_URL, EMBEDDING_REQUEST_ENDPOINT, COMPRESS_ENCODING_PARAMETER) :
				String.format("%s/%s?", DEFAULT_SERVICE_URL, EMBEDDING_REQUEST_ENDPOINT);

		HttpPost post = new HttpPost(url);
		post.setEntity(entityBuilder.build());
		return post;
	}

	public static OnnxTensor getImageEmbedding(OrtEnvironment ortEnvironment, CloseableHttpClient client, HttpPost post) throws IOException, OrtException, HttpException {

		try (var response = client.execute(post)) {
			switch (response.getStatusLine().getStatusCode()) {
			case HTTP_CANCELLED:
				throw new RuntimeException("Cancelled Embedding Request");

			case HTTP_SUCCESS: {
				byte[] responseEntity = EntityUtils.toByteArray(response.getEntity());
				byte[] decodedEmbedding = Base64.getDecoder().decode(responseEntity);
				ByteBuffer directBuffer = ByteBuffer.allocateDirect(decodedEmbedding.length).order(ByteOrder.nativeOrder());
				directBuffer.put(decodedEmbedding, 0, decodedEmbedding.length);
				directBuffer.position(0);
				FloatBuffer floatBuffEmbedding = directBuffer.asFloatBuffer();
				floatBuffEmbedding.position(0);
				final OnnxTensor embedding = OnnxTensor.createTensor(ortEnvironment, floatBuffEmbedding, new long[]{1, 256, 64, 64});
				return embedding;
			}

			default: {
				if (response.getEntity() != null) {
					throw new HttpException(EntityUtils.toString(response.getEntity()));
				} else {
					throw new HttpException("Received Error Code: " + response.getStatusLine().getStatusCode());
				}
			}
			}
		}
	}

	public static long otsuThresholdPrediction(long[] histogram) {

		long histogramIntensity = 0;
		long numPoints = 0;

		for (int idx = 0; idx < histogram.length; idx++) {
			histogramIntensity += idx * histogram[idx];
			numPoints += histogram[idx];
		}

		long intensitySumBelowThreshold = 0;
		long numPointsBelowThreshold = histogram[0];

		double interClassVariance;
		double maxInterClassVariance = 0.0;
		int predictedThresholdIdx = 0;

		for (int i = 1; i < histogram.length - 1; i++) {
			intensitySumBelowThreshold += i * histogram[i];
			numPointsBelowThreshold += histogram[i];

			double denom = (double)numPointsBelowThreshold * (numPoints - numPointsBelowThreshold);

			if (denom != 0.0) {
				double num = ((double)numPointsBelowThreshold / numPoints) * histogramIntensity - intensitySumBelowThreshold;
				interClassVariance = Math.pow(num, 2) / denom;
			} else {
				interClassVariance = 0.0;
			}

			if (interClassVariance >= maxInterClassVariance) {
				maxInterClassVariance = interClassVariance;
				predictedThresholdIdx = i;
			}
		}
		return predictedThresholdIdx;
	}

	public static double calculateThreshold(RandomAccessibleInterval<FloatType> prediction) {
		/* [-40..30] seems from testing like a reasonable range to include the vast majority of
		 *  prediction values, excluding perhaps some extreme outliers (which imo is desirable) */
		Real1dBinMapper<FloatType> binMapper = new Real1dBinMapper<>(-40.0, 30.0, 256, false);
		long[] histogram = new long[(int)binMapper.getBinCount()];

		LoopBuilder.setImages(prediction).forEachPixel(pixel -> {
			int binIdx = (int)binMapper.map(pixel);
			if (binIdx != -1)
				histogram[binIdx]++;
		});

		FloatType binVar = new FloatType();

		long otsuIdx = otsuThresholdPrediction(histogram);
		binMapper.getUpperBound((int)otsuIdx, binVar);

		return binVar.getRealDouble();
	}

	public static Map<String, OnnxTensor> noMaskParameters(OrtEnvironment environment) throws OrtException {

		return Map.of(
				MASK_INPUT, OnnxTensor.createTensor(environment, noMaskBuffer, maskShape),
				HAS_MASK_INPUT, OnnxTensor.createTensor(environment, hasNoMaskInput, hasMaskFlagShape)
		);
	}

	public static Map<String, OnnxTensor> pointParameters(OrtEnvironment env, Point point) throws OrtException {

		long numPoints = 1;

		FloatBuffer coordsBuffer = allocateDirectFloatBuffer((int)(numPoints * 2));
		FloatBuffer labelsBuffer = allocateDirectFloatBuffer((int)numPoints);

		/* SAM wants points in the center of pixels */
		coordsBuffer.put(point.getFloatPosition(0) + 0.5f);
		coordsBuffer.put(point.getFloatPosition(1) + 0.5f);

		labelsBuffer.put(1f); // point IN

		coordsBuffer.position(0);
		labelsBuffer.position(0);

		OnnxTensor coords = OnnxTensor.createTensor(env, coordsBuffer, new long[]{1, numPoints, 2});
		OnnxTensor labels = OnnxTensor.createTensor(env, labelsBuffer, new long[]{1, numPoints});

		Map<String, OnnxTensor> params = new HashMap<>();
		params.put(POINT_COORDS, coords);
		params.put(POINT_LABELS, labels);
		return params;
	}

	public static FloatBuffer allocateDirectFloatBuffer(int size) {

		return ByteBuffer.allocateDirect(size * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
	}

	public static RandomAccessibleInterval<FloatType> getSamPrediction(OrtEnvironment env, OrtSession session, Point point, OnnxTensor embedding) throws OrtException {

		Map<String, OnnxTensor> predictionParameters = new HashMap<>();
		predictionParameters.putAll(pointParameters(env, point));
		predictionParameters.putAll(noMaskParameters(env));
		predictionParameters.putAll(sizeEmbeddingParameters(env, embedding));

		OnnxTensor masks = (OnnxTensor)session.run(predictionParameters).get(MASKS).get();
		RandomAccessibleInterval<FloatType> prediction = ArrayImgs.floats(masks.getFloatBuffer().array(), 1024, 1024);
		return prediction;
	}

	public static Map<String, OnnxTensor> sizeEmbeddingParameters(OrtEnvironment env, OnnxTensor embedding) throws OrtException {

		FloatBuffer imgSizeBuffer = ByteBuffer.allocateDirect(2 * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
		imgSizeBuffer.put(1024f).put(1024f);
		imgSizeBuffer.position(0);

		OnnxTensor imgSizeTensor = OnnxTensor.createTensor(env, imgSizeBuffer, new long[]{2});

		Map<String, OnnxTensor> params = new HashMap<>();
		params.put(ORIG_IM_SIZE, imgSizeTensor);
		params.put(IMAGE_EMBEDDINGS, embedding);
		return params;
	}

	public static List<RandomAccessibleInterval<BoolType>> getPredictionMasks(List<Point> points, OnnxTensor embedding, OrtEnvironment env, OrtSession session) throws OrtException {

		List<RandomAccessibleInterval<BoolType>> components = new ArrayList<>();
		for (Point point : points) {
			RandomAccessibleInterval<FloatType> prediction = getSamPrediction(env, session, point, embedding);

			double threshold = calculateThreshold(prediction) * 2;

			RandomAccessibleInterval<BoolType> predictionMask = Converters.convert(prediction, (input, output) -> {
				output.set(input.get() >= threshold);
			}, new BoolType());

			RandomAccessibleInterval<UnsignedLongType> connectedComponents = ArrayImgs.unsignedLongs(Intervals.dimensionsAsLongArray(prediction));
			ConnectedComponents.labelAllConnectedComponents(
					predictionMask,
					connectedComponents,
					ConnectedComponents.StructuringElement.FOUR_CONNECTED
			);

			long pointComponent = connectedComponents.randomAccess().setPositionAndGet(point).get();

			RandomAccessibleInterval<BoolType> singleComponentImg = Converters.convert(connectedComponents, (input, output) -> {
				output.set(input.get() == pointComponent);
			}, new BoolType());

			components.add(singleComponentImg);
		}
		return components;
	}

	public static void main(String[] args) throws IOException, OrtException, HttpException {

		final ArrayList<Point> points = createTestPoints();
		final RandomAccessibleInterval<IntType> img = createTestImg(points);
		ImageJFunctions.show(img, "img");

		OrtEnvironment env = OrtEnvironment.getEnvironment();
		CloseableHttpClient client = createSamServiceClient();

		HttpPost post = createSamEmbeddingPostRequest(
				raiToPng(img),
				null,
				false
		);

		OnnxTensor embedding = getImageEmbedding(env, client, post);
		OrtSession session = createOrtSession(env);

		final List<RandomAccessibleInterval<BoolType>> masks = getPredictionMasks(points, embedding, env, session);


		/* Show as a composite */
		final ArrayImg<IntType, IntArray> composite = ArrayImgs.ints(1024, 1024);
		final Cursor<IntType> cursor = Views.flatIterable(composite).cursor();
		while (cursor.hasNext()) {
			cursor.next();
			final Point point = cursor.positionAsPoint();
			for (int i = 0; i < masks.size(); i++) {
				final boolean inside = masks.get(i).getAt(point).get();
				if (inside) {
					cursor.get().set(i+1);
					break;
				}
			}
		}
		ImageJFunctions.show(composite, "composite");
	}

	private static ArrayList<Point> createTestPoints() {

		final Random random = new Random();
		final ArrayList<Point> points = new ArrayList<>();
		final int numPoints = 5;
		final int rowHeight = (1024 / numPoints);
		for (int i = 0; i < numPoints; i++) {
			final int centerX = random.nextInt(0, 1024);

			final int centerY = i * rowHeight + rowHeight / 2;
			points.add(new Point(centerX, centerY));
		}
		return points;
	}

	public static RandomAccessibleInterval<IntType> createTestImg(List<Point> centers) {

		final Interval samInterval = Intervals.createMinSize(0, 0, 1024, 1024);
		IntervalView<IntType> img = Views.interval(ArrayImgs.ints(1024, 1024), samInterval);
		RandomAccess<IntType> access = img.randomAccess();

		for (int x = 0; x < 1024; x++) {
			for (int y = 0; y < 1024; y++) {
				double closestDistance = Double.MAX_VALUE;
				for (Point center : centers) {
					double distanceFromCenter = Math.sqrt(Math.pow((x - center.getIntPosition(0)), 2.0) + Math.pow((y - center.getIntPosition(1)), 2.0));
					if (distanceFromCenter < closestDistance) {
						closestDistance = distanceFromCenter;
					}
				}
				access.setPositionAndGet(x, y).set((int)closestDistance);
			}
		}
		return img;
	}
}
