#pragma once

#include <string>
#include <array>
#include <iostream>
#include "Vector3.cuh.cu"

namespace RayTracing
{

struct Trajectory
{

float r, z, phi;

float rA, zA;

float rOm, zOm, phiOm;

float rP, zP;

}; // Trajectory

struct FigureData
{

RayTracing::Vector3 origin;

RayTracing::Color color;

float radius;

float reflectance, transparency;

int edgeLightsNum;

}; // FigureData

struct FloorData
{

RayTracing::Vector3 A, B, C, D;

std::string texturePath;

RayTracing::Color color;

float reflectance;

}; // FloorData

struct LightSourceData
{

RayTracing::Vector3 origin;

float radius;

RayTracing::Color color;

}; // LightSource

struct Config
{

int framesNum;

char outputTemplate[1024];

int width;
int height;

float horizontalViewDegrees;

Trajectory lookFrom, lookAt;

FigureData A, B, C;

FloorData floorData;

int lightSourcesNum;

std::array<LightSourceData, 4> lightSources;

int recursionDepth;

float samplesPerPixel;

}; // Config


std::istream& operator>>(std::istream &istream, Config& config);

std::istream& operator>>(std::istream &istream, Trajectory& trajectory);

std::istream& operator>>(std::istream &istream, FigureData& figureData);

std::istream& operator>>(std::istream &istream, FloorData& floorData);

std::istream& operator>>(std::istream &istream, LightSourceData& lightSourceData);

} // namespace RayTracing
