#include "helpers.hpp"

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/fitting/nonlinear_camera_estimation.hpp"
#include "eos/render/utils.hpp"

#include "rcr/model.hpp"
#include "cereal/cereal.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

void myDrawText(cv::Mat & image, const std::string & text, cv::Point loc)
{
  cv::Scalar black(0, 0, 0);
  cv::Scalar white(180, 180, 180);
  const double fontSize = 0.4;

  for (int x = - 1; x <= 1; x++)
    for (int y = - 1; y <= 1; y++)
    {
      cv::Point loc2(loc.x +x, loc.y + y);
      cv::putText(image, text, loc2, cv::FONT_HERSHEY_COMPLEX, fontSize, black);
    }
  cv::putText(image, text, loc, cv::FONT_HERSHEY_COMPLEX, fontSize, white);
}

void myDrawLandmarks(cv::Mat & image, const rcr::LandmarkCollection<cv::Vec2f> & landmarks)
{
  const int w = 2;
  for(const auto &landmark : landmarks)
  {
    cv::Point loc((int)landmark.coordinates[0], (int)landmark.coordinates[1]);
    cv::rectangle(image, cv::Point(loc.x - w, loc.y - w), cv::Point(loc.x + w, loc.y + w), cv::Scalar(255, 255, 0));
    myDrawText(image, landmark.name, cv::Point(loc.x + w + 2, loc.y));
  }
  myDrawText(image, "Press R to reset face", cv::Point(10, 10));
}

int main(int argc, char *argv[])
{
  rcr::detection_model rcr_model;
  // Load the landmark detection model:
  try {
    rcr_model = rcr::load_detection_model("../share/face_landmarks_model_rcr_68.bin");
  }
  catch (const cereal::Exception& e) {
    std::cout << "Error reading the RCR model " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  // Load the face detector from OpenCV:
  cv::CascadeClassifier face_cascade;
  if (!face_cascade.load("../share/haarcascade_frontalface_alt2.xml"))
  {
    std::cout << "Error loading the face detector" << std::endl;
    return EXIT_FAILURE;
  }

  cv::VideoCapture cap;
  cap.open(0); //open the default camera
  if (!cap.isOpened()) {
    std::cout << "Couldn't open the given file or camera 0." << std::endl;
    return EXIT_FAILURE;
  }


  cv::namedWindow("video", 1);

  cv::Mat frame, unmodified_frame;

  bool have_face = false;
  rcr::LandmarkCollection<cv::Vec2f> current_landmarks;
  cv::Rect current_facebox;

  for (;;)
  {
    cap >> frame; // get a new frame from camera
    if (frame.empty()) { // stop if we're at the end of the video
      break;
    }

    // We do a quick check if the current face's width is <= 50 pixel.
    // If it is, we re-initialise the tracking with the face detector.
    if (have_face && get_enclosing_bbox(rcr::to_row(current_landmarks)).width <= 50) {
      std::cout << "Reinitialising because the face bounding-box width is <= 50 px" << std::endl;
      have_face = false;
    }

    unmodified_frame = frame.clone();

    if (!have_face) {

      // Détection sur image initiale

      // Run the face detector and obtain the initial estimate using the mean landmarks:
      std::vector<cv::Rect> detected_faces;
      face_cascade.detectMultiScale(unmodified_frame, detected_faces, 1.2, 2, 0, cv::Size(110, 110));
      if (detected_faces.empty()) {
        cv::imshow("video", frame);
        cv::waitKey(30);
        continue;
      }
      cv::rectangle(frame, detected_faces[0], { 255, 0, 0 });
      // Rescale the V&J facebox to make it more like an ibug-facebox:
      // (also make sure the bounding box is square, V&J's is square)
      cv::Rect ibug_facebox = rescale_facebox(detected_faces[0], 0.85, 0.2);

      current_landmarks = rcr_model.detect(unmodified_frame, ibug_facebox);
      //rcr::draw_landmarks(frame, current_landmarks);
      myDrawLandmarks(frame, current_landmarks);

      have_face = true;
    }
    else {
      //
      // Detection sur images suivantes
      //

      // We already have a face - track and initialise using the enclosing bounding
      // box from the landmarks from the last frame:
      auto enclosing_bbox = get_enclosing_bbox(rcr::to_row(current_landmarks));
      enclosing_bbox = make_bbox_square(enclosing_bbox);
      current_landmarks = rcr_model.detect(unmodified_frame, enclosing_bbox);
      //rcr::draw_landmarks(frame, current_landmarks, { 0, 255, 0 }); // green, the new optimised landmarks
      myDrawLandmarks(frame, current_landmarks);
    }


    cv::imshow("video", frame);
    auto key = cv::waitKey(30);
    if (key == 'q') break;
    if (key == 'r') {
      have_face = false;
    }
  }

  return EXIT_SUCCESS;
};

