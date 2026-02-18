use actix_multipart::Multipart;
use actix_web::{post, web, App, HttpResponse, HttpServer, Responder};
use futures_util::TryStreamExt;
use image::io::Reader as ImageReader;
use image::imageops::FilterType;
use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::io::Cursor;
use std::sync::Mutex; // FIX 1: Import Mutex

struct AppState {
    // FIX 2: Wrap Session in a Mutex to allow safe mutable access
    session: Mutex<Session>,
}

fn preprocess(image_bytes: &[u8]) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let img = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()?
        .decode()?;

    let resized = img.resize_exact(224, 224, FilterType::Triangle);

    let mut input_tensor = Array4::<f32>::zeros((1, 3, 224, 224));
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    for (x, y, pixel) in resized.to_rgb8().enumerate_pixels() {
        let r = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
        let g = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
        let b = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];

        input_tensor[[0, 0, y as usize, x as usize]] = r;
        input_tensor[[0, 1, y as usize, x as usize]] = g;
        input_tensor[[0, 2, y as usize, x as usize]] = b;
    }

    Ok(input_tensor)
}

#[post("/predict")]
async fn predict(body: web::Bytes, data: web::Data<AppState>) -> impl Responder {
    // 1. Convert web::Bytes directly into a Vec or slice
    let image_bytes = body.to_vec();

    // 2. Preprocess (same as before)
    let input_array = match preprocess(&image_bytes) {
        Ok(t) => t,
        Err(_) => return HttpResponse::BadRequest().body("Invalid image data"),
    };

    let input_tensor = Tensor::from_array((
        [1, 3, 224, 224], 
        input_array.into_raw_vec()
    )).unwrap();

    // 3. Inference with Mutex lock
    let mut session = data.session.lock().unwrap();
    let outputs = session.run(ort::inputs![input_tensor]).unwrap();
    
    let (_, extracted_data) = outputs[0].try_extract_tensor::<f32>().unwrap();
    
    let mut best_index = 0;
    let mut best_score = -1.0;
    
    for (i, &score) in extracted_data.iter().enumerate() {
        if score > best_score {
            best_score = score;
            best_index = i;
        }
    }

    HttpResponse::Ok().json(serde_json::json!({
        "class_index": best_index,
        "confidence": best_score,
        "message": "Rust (ORT 2.0) is blazing fast!"
    }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Loading Rust Model...");
    
    let session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .commit_from_file("../models/mobilenetv2.onnx")
        .unwrap();
    
    println!("Model Loaded! Starting Server at 0.0.0.0:8000");

    // FIX 4: Wrap the session in Mutex::new()
    let state = web::Data::new(AppState { 
        session: Mutex::new(session) 
    });

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .service(predict)
    })
    .bind(("0.0.0.0", 8000))?
    .run()
    .await
}